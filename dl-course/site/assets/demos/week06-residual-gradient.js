DL.register('week06-residual-gradient', function (root) {
  var D = 48, LMAX = 100, NS = 4;
  var rand = DL.rng(2024), Z = [], X = [];
  for (var l = 0; l < LMAX; l++) { var m = new Float64Array(D * D); for (var i = 0; i < D * D; i++) m[i] = DL.randn(rand); Z.push(m); }
  for (var s = 0; s < NS; s++) { var x = new Float64Array(D); for (var k = 0; k < D; k++) x[k] = DL.randn(rand); X.push(x); }
  var G0 = []; for (var s2 = 0; s2 < NS; s2++) { var g = new Float64Array(D); for (var q = 0; q < D; q++) g[q] = DL.randn(rand) / Math.sqrt(D); G0.push(g); }

  // Returns log10 of ||dL/dh_l|| / ||dL/dh_L|| for l = 0..L, averaged (in squares) over NS samples
  function gradProfile(L, gain, branch, residual) {
    var acc = new Float64Array(L + 1), sc = Math.sqrt(2 / D) * gain * (residual ? branch : 1);
    for (var s = 0; s < NS; s++) {
      var h = X[s], masks = [];
      for (var l = 0; l < L; l++) {
        var W = Z[l], z = new Float64Array(D), mk = new Uint8Array(D);
        for (var i = 0; i < D; i++) { var t = 0, row = i * D; for (var j = 0; j < D; j++) t += W[row + j] * h[j]; t *= sc; z[i] = t > 0 ? t : 0; mk[i] = t > 0 ? 1 : 0; }
        var hn = new Float64Array(D);
        for (var a = 0; a < D; a++) hn[a] = residual ? h[a] + z[a] : z[a];
        masks.push(mk); h = hn;
        // forward overflow guard
        var nn = 0; for (var b = 0; b < D; b++) nn += h[b] * h[b];
        if (!isFinite(nn) || nn > 1e200) { for (var c2 = 0; c2 < D; c2++) h[c2] = 0; }
      }
      var gcur = G0[s], n0 = Math.sqrt(gcur.reduce(function (p, v) { return p + v * v; }, 0));
      acc[L] += 1;
      for (var ll = L - 1; ll >= 0; ll--) {
        var Wl = Z[ll], mk2 = masks[ll], gn = new Float64Array(D);
        for (var jj = 0; jj < D; jj++) gn[jj] = residual ? gcur[jj] : 0;
        for (var ii = 0; ii < D; ii++) { if (!mk2[ii]) continue; var gi = gcur[ii] * sc, ro = ii * D; for (var j2 = 0; j2 < D; j2++) gn[j2] += Wl[ro + j2] * gi; }
        gcur = gn;
        var nr = 0; for (var u = 0; u < D; u++) nr += gcur[u] * gcur[u];
        acc[ll] += nr / (n0 * n0);
      }
    }
    var out = [];
    for (var e = 0; e <= L; e++) out.push(0.5 * Math.log10(Math.max(1e-300, e === L ? 1 : acc[e] / NS)));
    return out;
  }

  var ctrl = DL.controls(root);
  var depth = DL.slider(ctrl, { label: 'Depth L (layers)', min: 4, max: LMAX, step: 1, value: 50 }, draw);
  var gain = DL.slider(ctrl, { label: 'Weight gain (1 = He init)', min: 0.5, max: 1.5, step: 0.01, value: 0.9, format: function (v) { return (+v).toFixed(2); } }, draw);
  var br = DL.slider(ctrl, { label: 'Residual branch scale', min: 0, max: 1, step: 0.01, value: 0.1, format: function (v) { return (+v).toFixed(2); } }, draw);
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Gradient norm at each layer, relative to the loss end (right). Plain network: the gradient is multiplied by a random Jacobian at every layer, so a gain of 0.9 instead of 1 compounds to 0.9^L-ish decay and 1.1 explodes. Residual network: every layer adds the identity, so a gradient path of length zero always exists; shrink the branch scale (like zero-initialising the last BN gamma) to keep it stable.' }));
  function draw() {
    if (!c.w) return;
    var L = Math.round(depth.get()), col = DL.colors();
    var p = gradProfile(L, gain.get(), 1, false), r = gradProfile(L, gain.get(), br.get(), true);
    var xs = DL.range(L + 1), all = p.concat(r).filter(isFinite);
    var lo = Math.min(-1, Math.floor(Math.min.apply(null, all))), hi = Math.max(1, Math.ceil(Math.max.apply(null, all)));
    var P = DL.plot(c, { xmin: 0, xmax: L, ymin: lo, ymax: hi, xlabel: 'layer index (0 = input side, L = loss side)', ylabel: 'log10 of gradient norm ratio', pad: { l: 52 } });
    P.clear(); P.grid(); P.axes(); P.hline(0, col.muted);
    P.line(xs, p, col.red, 2.5); P.line(xs, r, col.blue, 2.5);
    P.text('plain', L * 0.04, hi - (hi - lo) * 0.07, col.red, 'left'); P.text('residual', L * 0.04, hi - (hi - lo) * 0.14, col.blue, 'left');
    var f = function (v) { return Math.abs(v) < 6 ? '10^' + DL.fmt(v, 1) : '10^' + DL.fmt(v, 0); };
    say('Gradient at layer 0 relative to the top: plain <b>' + f(p[0]) + '</b>, residual <b>' + f(r[0]) + '</b>. Unrolled, the residual net contains 2^' + L + ' = 10^' + DL.fmt(L * Math.log10(2), 1) + ' paths; the shortest has length 0.' + (p[0] < -6 ? ' <b>Plain: vanishing gradient.</b>' : p[0] > 6 ? ' <b>Plain: exploding gradient.</b>' : ''));
  }
  c.draw = draw;
});
