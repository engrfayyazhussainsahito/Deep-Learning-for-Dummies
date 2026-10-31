DL.register('week12-kl-gaussians', function (root) {
  var ctrl = DL.controls(root), row = DL.row(root);
  var m1 = DL.slider(ctrl, { label: 'μ_p', min: -4, max: 4, step: 0.05, value: 0 }, draw);
  var s1 = DL.slider(ctrl, { label: 'σ_p', min: 0.2, max: 3, step: 0.05, value: 1 }, draw);
  var m2 = DL.slider(ctrl, { label: 'μ_q', min: -4, max: 4, step: 0.05, value: 1 }, draw);
  var s2 = DL.slider(ctrl, { label: 'σ_q', min: 0.2, max: 3, step: 0.05, value: 1.5 }, draw);
  var tgt = DL.select(ctrl, { label: 'Target p', value: 'gauss', options: [{ value: 'gauss', label: 'single Gaussian N(μ_p, σ_p²)' }, { value: 'mix', label: 'bimodal mixture (modes at ±2.5, σ=0.6)' }] }, draw);
  DL.button(row, 'q ← argmin KL(p‖q)  (forward)', function () { fit('fwd'); });
  DL.button(row, 'q ← argmin KL(q‖p)  (reverse)', function () { fit('rev'); });
  DL.button(row, 'reset q', function () { m2.set(1); s2.set(1.5); draw(); });
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'With a single Gaussian target, compare the numeric KL with the closed form and watch KL(p‖q) and KL(q‖p) differ when σ_p ≠ σ_q. Switch to the bimodal target and press the two fit buttons: forward KL (the maximum-likelihood direction) spreads q over both modes and puts mass between them; reverse KL (the VAE posterior direction) locks onto one mode. Shaded area: the integrand of KL(p‖q).' }));
  var XMIN = -8, XMAX = 8, NG = 801, xs = DL.linspace(XMIN, XMAX, NG), dx = (XMAX - XMIN) / (NG - 1);
  function gauss(x, m, s) { return Math.exp(-0.5 * (x - m) * (x - m) / (s * s)) / (s * 2.5066282746); }
  function pden(x) { return tgt.get() === 'mix' ? 0.5 * gauss(x, -2.5, 0.6) + 0.5 * gauss(x, 2.5, 0.6) : gauss(x, m1.get(), s1.get()); }
  function qden(x, m, s) { return gauss(x, m, s); }
  var E = 1e-300;
  function klNum(pa, qa) { var s = 0; for (var i = 0; i < NG; i++) if (pa[i] > 1e-12) s += pa[i] * Math.log(pa[i] / Math.max(qa[i], E)) * dx; return s; }
  function arr(f) { return xs.map(f); }
  function fit(kind) {
    var pa = arr(pden), best = null, bm = 0, bs = 1, mu, sg;
    if (kind === 'fwd') {
      var mean = 0, v = 0, i;
      for (i = 0; i < NG; i++) mean += xs[i] * pa[i] * dx;
      for (i = 0; i < NG; i++) v += (xs[i] - mean) * (xs[i] - mean) * pa[i] * dx;
      m2.set(DL.clamp(mean, -4, 4)); s2.set(DL.clamp(Math.sqrt(v), 0.2, 3));
    } else {
      for (mu = -4; mu <= 4.001; mu += 0.1) for (sg = 0.2; sg <= 3.001; sg += 0.05) {
        var qa = arr(function (x) { return qden(x, mu, sg); });
        var k = klNum(qa, pa);
        if (best === null || k < best) { best = k; bm = mu; bs = sg; }
      }
      m2.set(bm); s2.set(bs);
    }
    draw();
  }
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), ctx = c.ctx, mu = m2.get(), sg = s2.get();
    var pa = arr(pden), qa = arr(function (x) { return qden(x, mu, sg); });
    var ymax = Math.max(0.5, Math.max.apply(null, pa) * 1.15, Math.max.apply(null, qa) * 1.15);
    var P = DL.plot(c, { xmin: XMIN, xmax: XMAX, ymin: -0.15 * ymax, ymax: ymax, xlabel: 'x', ylabel: 'density' });
    P.clear(); P.grid(); P.axes();
    // shade integrand p log(p/q), scaled for visibility (clamped at 0 for display, sign shown by colour)
    ctx.save(); ctx.globalAlpha = 0.35;
    for (var i = 0; i < NG - 1; i += 2) {
      if (pa[i] < 1e-8) continue;
      var t = pa[i] * Math.log(pa[i] / Math.max(qa[i], E));
      var h = DL.clamp(Math.abs(t), 0, ymax * 0.15);
      ctx.fillStyle = t >= 0 ? col.red : col.green;
      var x0 = P.px(xs[i]), x1 = P.px(xs[Math.min(i + 2, NG - 1)]), y0 = P.py(0), y1 = P.py(-h);
      ctx.fillRect(x0, y0, Math.max(1, x1 - x0), y1 - y0);
    }
    ctx.restore();
    P.line(xs, pa, col.blue, 2.5); P.line(xs, qa, col.accent, 2.5);
    P.text('p (target)', XMIN + 0.4, ymax * 0.93, col.blue, 'left'); P.text('q (model)', XMIN + 0.4, ymax * 0.84, col.accent, 'left');
    var klpq = klNum(pa, qa), klqp = klNum(qa, pa);
    var mix = arr(function (x, k) { return 0; });
    var js = 0; for (var j = 0; j < NG; j++) { var mm = 0.5 * (pa[j] + qa[j]); if (pa[j] > 1e-12) js += 0.5 * pa[j] * Math.log(pa[j] / mm) * dx; if (qa[j] > 1e-12) js += 0.5 * qa[j] * Math.log(qa[j] / mm) * dx; }
    var txt = 'KL(p‖q) = <b>' + DL.fmt(klpq, 3) + '</b> &nbsp; KL(q‖p) = <b>' + DL.fmt(klqp, 3) + '</b> &nbsp; JS(p,q) = <b>' + DL.fmt(js, 3) + '</b> (max ln 2 = 0.693)';
    if (tgt.get() === 'gauss') {
      var a = m1.get(), b = s1.get();
      var cf = Math.log(sg / b) + (b * b + (a - mu) * (a - mu)) / (2 * sg * sg) - 0.5;
      txt += '<br>closed form KL(p‖q) = ln(σ_q/σ_p) + (σ_p² + (μ_p−μ_q)²)/(2σ_q²) − ½ = <b>' + DL.fmt(cf, 3) + '</b> &nbsp; W₂(p,q) = √((Δμ)² + (Δσ)²) = <b>' + DL.fmt(Math.sqrt((a - mu) * (a - mu) + (b - sg) * (b - sg)), 3) + '</b>';
    }
    say(txt);
  }
  c.draw = draw;
});
