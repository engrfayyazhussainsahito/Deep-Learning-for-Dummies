DL.register('week12-gan-playground', function (root) {
  // 1-D GAN on a Gaussian mixture, trained live in the browser with hand-written backprop.
  var ctrl = DL.controls(root), row = DL.row(root);
  var lossSel = DL.select(ctrl, { label: 'Objective', value: 'ns', options: [
    { value: 'ns', label: 'non-saturating GAN' }, { value: 'mm', label: 'minimax GAN (saturating)' }, { value: 'wgan', label: 'WGAN (weight clipping)' }] }, reset);
  var dataSel = DL.select(ctrl, { label: 'Data', value: '4', options: [{ value: '1', label: '1 mode' }, { value: '2', label: '2 modes' }, { value: '4', label: '4 modes' }] }, reset);
  var lrS = DL.slider(ctrl, { label: 'Learning rate', min: -4, max: -1.5, step: 0.05, value: -2.6, format: function (v) { return DL.fmt(Math.pow(10, v), 4); } }, null);
  var kS = DL.slider(ctrl, { label: 'D steps per G step', min: 1, max: 10, step: 1, value: 1 }, null);
  var playBtn = DL.button(row, 'Play', toggle);
  DL.button(row, 'Step ×50', function () { run(50); draw(); });
  DL.button(row, 'Reset', reset);
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Blue: real data density. Red: histogram of generator samples. Green: discriminator output (probability of real) or the critic value (WGAN, rescaled). Try the 4-mode data with 1 D step per G step and watch the generator hop between modes; then try 5 D steps, or WGAN. Results vary with the random seed (Reset re-seeds); this is a toy, so treat it as an illustration of the dynamics, not a benchmark.' }));
  var H = 16, rnd, seedN = 1, step = 0, playing = false, raf = 0, visible = true, lastD = 0, lastG = 0, gaussCount = 0;
  function G() { return DL.randn(rnd); }
  function Net(sizes) {
    var n = { W: [], b: [], mW: [], vW: [], mb: [], vb: [], t: 0 };
    for (var l = 0; l < sizes.length - 1; l++) {
      var W = [], mW = [], vW = [], i, j;
      for (i = 0; i < sizes[l + 1]; i++) { var r = [], z = [], z2 = []; for (j = 0; j < sizes[l]; j++) { r.push(G() * Math.sqrt(1 / sizes[l])); z.push(0); z2.push(0); } W.push(r); mW.push(z); vW.push(z2); }
      n.W.push(W); n.mW.push(mW); n.vW.push(vW);
      var zb = function (k) { var a = []; for (var q = 0; q < k; q++) a.push(0); return a; };
      n.b.push(zb(sizes[l + 1])); n.mb.push(zb(sizes[l + 1])); n.vb.push(zb(sizes[l + 1]));
    }
    return n;
  }
  function fwd(n, x) { // x: array of inputs (batch of vectors); returns cache
    var acts = [x], a = x;
    for (var l = 0; l < n.W.length; l++) {
      var out = a.map(function (v) { return n.W[l].map(function (row, i) { var s = n.b[l][i]; for (var j = 0; j < row.length; j++) s += row[j] * v[j]; return l < n.W.length - 1 ? Math.tanh(s) : s; }); });
      acts.push(out); a = out;
    }
    return acts;
  }
  function bwd(n, acts, dOut) { // returns {gW, gb, dIn}
    var L = n.W.length, gW = [], gb = [], d = dOut, B = acts[0].length;
    for (var l = L - 1; l >= 0; l--) {
      var gw = n.W[l].map(function (r) { return r.map(function () { return 0; }); }), gbb = n.b[l].map(function () { return 0; }), dprev = [];
      for (var k = 0; k < B; k++) {
        var dz = d[k].slice();
        if (l < L - 1) for (var i = 0; i < dz.length; i++) dz[i] *= (1 - acts[l + 1][k][i] * acts[l + 1][k][i]);
        var dp = acts[l][k].map(function () { return 0; });
        for (i = 0; i < dz.length; i++) { gbb[i] += dz[i]; for (var j = 0; j < dp.length; j++) { gw[i][j] += dz[i] * acts[l][k][j]; dp[j] += dz[i] * n.W[l][i][j]; } }
        dprev.push(dp);
      }
      gW[l] = gw; gb[l] = gbb; d = dprev;
    }
    return { gW: gW, gb: gb, dIn: d };
  }
  function adam(n, g, lr, b1) {
    n.t++; var t = n.t, B2 = 0.999;
    for (var l = 0; l < n.W.length; l++) {
      for (var i = 0; i < n.W[l].length; i++) {
        for (var j = 0; j < n.W[l][i].length; j++) {
          var gg = g.gW[l][i][j]; n.mW[l][i][j] = b1 * n.mW[l][i][j] + (1 - b1) * gg; n.vW[l][i][j] = B2 * n.vW[l][i][j] + (1 - B2) * gg * gg;
          n.W[l][i][j] -= lr * (n.mW[l][i][j] / (1 - Math.pow(b1, t))) / (Math.sqrt(n.vW[l][i][j] / (1 - Math.pow(B2, t))) + 1e-8);
        }
        var gb = g.gb[l][i]; n.mb[l][i] = b1 * n.mb[l][i] + (1 - b1) * gb; n.vb[l][i] = B2 * n.vb[l][i] + (1 - B2) * gb * gb;
        n.b[l][i] -= lr * (n.mb[l][i] / (1 - Math.pow(b1, t))) / (Math.sqrt(n.vb[l][i] / (1 - Math.pow(B2, t))) + 1e-8);
      }
    }
  }
  function clip(n, c) { n.W.forEach(function (W) { W.forEach(function (r) { for (var j = 0; j < r.length; j++) r[j] = DL.clamp(r[j], -c, c); }); }); }
  var gen, dis;
  function centers() { var m = dataSel.get(); return m === '1' ? [0] : (m === '2' ? [-2, 2] : [-3, -1, 1, 3]); }
  function realBatch(B) { var cs = centers(), a = []; for (var i = 0; i < B; i++) a.push([cs[Math.floor(rnd() * cs.length)] + 0.2 * G()]); return a; }
  function noise(B) { var a = []; for (var i = 0; i < B; i++) a.push([G()]); return a; }
  function reset() {
    seedN++; rnd = DL.rng(seedN * 7919); step = 0; lastD = 0; lastG = 0;
    gen = Net([1, H, H, 1]); dis = Net([1, H, H, 1]);
    pause(); draw();
  }
  function run(n) {
    var B = 64, lr = Math.pow(10, lrS.get()), mode = lossSel.get(), K = Math.round(kS.get());
    for (var s = 0; s < n; s++) {
      for (var k = 0; k < K; k++) {
        var xr = realBatch(B), xf = fwd(gen, noise(B))[3];
        var ar = fwd(dis, xr), af = fwd(dis, xf), dr = [], df = [];
        for (var i = 0; i < B; i++) {
          var lr_ = ar[3][i][0], lf = af[3][i][0];
          if (mode === 'wgan') { dr.push([-1 / B]); df.push([1 / B]); }
          else { dr.push([(DL.sigmoid(lr_) - 1) / B]); df.push([DL.sigmoid(lf) / B]); }
        }
        var g1 = bwd(dis, ar, dr), g2 = bwd(dis, af, df);
        var tot = { gW: g1.gW.map(function (W, l) { return W.map(function (r, i) { return r.map(function (v, j) { return v + g2.gW[l][i][j]; }); }); }), gb: g1.gb.map(function (b, l) { return b.map(function (v, i) { return v + g2.gb[l][i]; }); }) };
        adam(dis, tot, mode === 'wgan' ? lr * 0.5 : lr, mode === 'wgan' ? 0 : 0.5);
        if (mode === 'wgan') clip(dis, 0.1);
        lastD = 0; for (i = 0; i < B; i++) lastD += mode === 'wgan' ? (ar[3][i][0] - af[3][i][0]) / B : -(Math.log(DL.sigmoid(ar[3][i][0]) + 1e-12) + Math.log(1 - DL.sigmoid(af[3][i][0]) + 1e-12)) / B;
      }
      var gz = fwd(gen, noise(B)), fx = gz[3], ad = fwd(dis, fx), dg = [];
      for (i = 0; i < B; i++) {
        var l2 = ad[3][i][0];
        if (mode === 'wgan') dg.push([-1 / B]);
        else if (mode === 'ns') dg.push([(DL.sigmoid(l2) - 1) / B]);   // minimise -log D(G(z))
        else dg.push([-DL.sigmoid(l2) / B]);                            // minimise log(1 - D(G(z)))
      }
      var bd = bwd(dis, ad, dg), gg = bwd(gen, gz, bd.dIn);
      adam(gen, gg, mode === 'wgan' ? lr * 0.5 : lr, mode === 'wgan' ? 0 : 0.5);
      step++;
    }
  }
  function toggle() { if (playing) pause(); else { playing = true; playBtn.textContent = 'Pause'; loop(); } }
  function pause() { playing = false; if (playBtn) playBtn.textContent = 'Play'; if (raf) cancelAnimationFrame(raf); raf = 0; }
  function loop() { if (!playing) return; if (visible) { run(3); draw(); } raf = requestAnimationFrame(loop); }
  if (window.IntersectionObserver) new IntersectionObserver(function (e) { visible = e[0].isIntersecting; if (!visible) pause(); }, {}).observe(root);
  function dens(x, cs) { var s = 0; cs.forEach(function (m) { s += Math.exp(-0.5 * (x - m) * (x - m) / 0.04) / (0.2 * 2.5066282746) / cs.length; }); return s; }
  function draw() {
    if (!c.w || !gen) return;
    var col = DL.colors(), ctx = c.ctx, cs = centers(), mode = lossSel.get();
    var P = DL.plot(c, { xmin: -5, xmax: 5, ymin: 0, ymax: 2.2, xlabel: 'x', ylabel: 'density' });
    P.clear(); P.grid(); P.axes();
    P.fn(function (x) { return dens(x, cs); }, col.blue, 2.5);
    var N = 2000, bins = 60, h = [], i, gs = fwd(gen, noise(N))[3].map(function (v) { return v[0]; });
    for (i = 0; i < bins; i++) h.push(0);
    gs.forEach(function (v) { var b = Math.floor((v + 5) / 10 * bins); if (b >= 0 && b < bins) h[b]++; });
    var bw = 10 / bins; ctx.save(); ctx.globalAlpha = 0.45; ctx.fillStyle = col.red;
    for (i = 0; i < bins; i++) { var d = h[i] / N / bw; ctx.fillRect(P.px(-5 + i * bw), P.py(Math.min(d, 2.2)), P.px(-5 + (i + 1) * bw) - P.px(-5 + i * bw) - 1, P.py(0) - P.py(Math.min(d, 2.2))); }
    ctx.restore();
    var xs = DL.linspace(-5, 5, 120), dv = fwd(dis, xs.map(function (x) { return [x]; }))[3].map(function (v) { return v[0]; });
    var ys = mode === 'wgan' ? dv.map(function (v) { return 1.1 + v * 3; }) : dv.map(function (v) { return DL.sigmoid(v) * 2; });
    P.line(xs, ys.map(function (y) { return DL.clamp(y, 0, 2.2); }), col.green, 2);
    P.text(mode === 'wgan' ? 'critic f(x) (rescaled)' : 'D(x) × 2', 3.4, 2.05, col.green, 'left');
    // mode coverage: modes with at least 3% of fake mass within 0.6
    var cov = 0; cs.forEach(function (m) { var k = 0; gs.forEach(function (v) { if (Math.abs(v - m) < 0.6) k++; }); if (k / N > 0.03) cov++; });
    gaussCount = cov;
    say('step <b>' + step + '</b> &nbsp; modes covered by generator: <b>' + cov + ' / ' + cs.length + '</b> &nbsp; ' + (mode === 'wgan' ? 'critic gap E f(real) − E f(fake) = ' : 'D loss = ') + '<b>' + DL.fmt(lastD, 3) + '</b>' + (cs.length > 1 && cov < cs.length && step > 400 ? ' &nbsp; <b>modes missing</b>' : ''));
  }
  c.draw = draw;
  reset();
});
