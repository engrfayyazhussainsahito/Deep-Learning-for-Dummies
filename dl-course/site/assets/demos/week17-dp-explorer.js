DL.register('week17-dp-explorer', function (root) {
  // Part 1 (left): Gaussian mechanism on a mean query. Two neighbouring datasets give two output distributions;
  //   the overlap shows how hard it is to tell them apart. The privacy loss bound is eps from sigma via the classical formula
  //   (and exact Gaussian-mechanism epsilon computed numerically for the given delta).
  // Part 2 (right): DP-SGD noise: clipped per-example gradients plus noise, signal-to-noise ratio of the averaged gradient.
  var ctrl = DL.controls(root);
  var sig = DL.slider(ctrl, { label: 'Noise multiplier σ', min: 0.2, max: 6, step: 0.05, value: 1.0, format: function (v) { return DL.fmt(v, 2); } }, draw);
  var delta = DL.slider(ctrl, { label: 'log10 δ', min: -9, max: -3, step: 1, value: -5, format: function (v) { return String(Math.round(v)); } }, draw);
  var bs = DL.slider(ctrl, { label: 'Batch size B', min: 16, max: 4096, step: 16, value: 256, format: function (v) { return String(Math.round(v)); } }, draw);
  var steps = DL.slider(ctrl, { label: 'Steps T', min: 10, max: 5000, step: 10, value: 500, format: function (v) { return String(Math.round(v)); } }, draw);
  var N = DL.slider(ctrl, { label: 'Dataset size N', min: 1000, max: 100000, step: 1000, value: 50000, format: function (v) { return String(Math.round(v)); } }, draw);
  var c = DL.canvas(root, { aspect: 0.42 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Left: the output density of a single-step Gaussian mechanism with sensitivity 1 on neighbouring datasets D (blue) and D′ (red, shifted by 1). More noise σ makes them overlap, so any observer learns less. Readouts: exact ε of that one release for the chosen δ, and a rough DP-SGD total using strong composition with subsampling q = B/N (an upper-bound heuristic, NOT a replacement for the moments / RDP accountant used in Opacus). Right: gradient signal versus noise per coordinate. Larger batches shrink the noise on the average gradient by 1/B.' }));

  function erf(x) { var s = x < 0 ? -1 : 1; x = Math.abs(x); var t = 1 / (1 + 0.3275911 * x), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; }
  function Phi(x) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
  // exact delta(eps) of the Gaussian mechanism with sensitivity 1 and noise sigma (Balle & Wang 2018)
  function deltaOf(eps, s) { var a = 1 / (2 * s) - eps * s, b = -1 / (2 * s) - eps * s; return Phi(a) - Math.exp(eps) * Phi(b); }
  function epsOf(s, d) {
    var lo = 0, hi = 200, i;
    if (deltaOf(0, s) <= d) return 0;
    for (i = 0; i < 80; i++) { var m = (lo + hi) / 2; if (deltaOf(m, s) > d) lo = m; else hi = m; }
    return hi;
  }
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), ctx = c.ctx, W = c.w, H = c.h, s = sig.get(), d = Math.pow(10, delta.get());
    ctx.clearRect(0, 0, W, H);
    var half = W / 2 - 10, pt = 22, pb = 30, ph = H - pt - pb, pl = 10;
    ctx.font = '11px sans-serif'; ctx.fillStyle = col.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('Gaussian mechanism: outputs on D and D′', pl + half / 2, pt - 6);
    ctx.fillText('DP-SGD: gradient signal vs noise (per step)', W / 2 + 10 + half / 2, pt - 6);
    // left: densities
    var xmin = -4 - 2 * s * 0, xmax = 5 + 3 * Math.min(s, 4), n = 200, i, ymax = 0, f0 = [], f1 = [];
    xmin = -3 - 2.5 * s; xmax = 4 + 2.5 * s;
    for (i = 0; i <= n; i++) { var x = xmin + (xmax - xmin) * i / n; f0.push(Math.exp(-x * x / (2 * s * s)) / (s * 2.5066)); f1.push(Math.exp(-(x - 1) * (x - 1) / (2 * s * s)) / (s * 2.5066)); ymax = Math.max(ymax, f0[i]); }
    function X(k) { return pl + (k / n) * half; }
    function Y(v) { return pt + ph - v / (ymax * 1.1) * ph; }
    [[f0, col.blue], [f1, col.red]].forEach(function (p) {
      ctx.beginPath(); ctx.moveTo(X(0), Y(0));
      for (i = 0; i <= n; i++) ctx.lineTo(X(i), Y(p[0][i]));
      ctx.lineTo(X(n), Y(0)); ctx.closePath(); ctx.fillStyle = p[1]; ctx.globalAlpha = 0.3; ctx.fill(); ctx.globalAlpha = 1;
      ctx.beginPath(); for (i = 0; i <= n; i++) { if (i) ctx.lineTo(X(i), Y(p[0][i])); else ctx.moveTo(X(i), Y(p[0][i])); } ctx.strokeStyle = p[1]; ctx.lineWidth = 2; ctx.stroke();
    });
    ctx.strokeStyle = col.line; ctx.beginPath(); ctx.moveTo(pl, pt + ph); ctx.lineTo(pl + half, pt + ph); ctx.stroke();
    // overlap area (total variation complement)
    var ov = 0; for (i = 0; i < n; i++) ov += Math.min(f0[i], f1[i]) * (xmax - xmin) / n;
    ctx.fillStyle = col.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('overlap = ' + DL.fmt(ov, 2), pl + 4, pt + 2);
    // right: signal vs noise bars
    var B = bs.get(), C = 1, signal = 1 * C, noiseStd = s * C / B, rx = W / 2 + 20, rw = half - 20;
    var maxv = Math.max(signal, noiseStd * 3, 1e-9);
    function bar(k, v, color, label) {
      var bw = rw / 3.2, x0 = rx + k * (bw + 14), h = Math.min(1, v / maxv) * (ph - 30);
      ctx.fillStyle = color; ctx.globalAlpha = 0.8; ctx.fillRect(x0, pt + ph - h, bw, h); ctx.globalAlpha = 1;
      ctx.fillStyle = col.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(label, x0 + bw / 2, pt + ph + 4);
      ctx.textBaseline = 'bottom'; ctx.fillText(v.toExponential(1), x0 + bw / 2, pt + ph - h - 2);
    }
    bar(0, 1, col.accent, 'clip norm C'); bar(1, noiseStd * Math.sqrt(1), col.amber, 'noise std σC/B'); bar(2, noiseStd * 3, col.red, '3x noise');
    // epsilon numbers
    var eps1 = epsOf(s, d), q = Math.min(1, B / N.get()), T = Math.round(steps.get());
    // heuristic: subsampled Gaussian per-step eps ~ 2 q sqrt(...) ; use strong composition on amplified per-step eps
    var epsStep = epsOf(s, d / (2 * T)), epsAmp = Math.log(1 + q * (Math.exp(epsStep) - 1));
    var dprime = d / 2;
    var epsTotal = epsAmp * Math.sqrt(2 * T * Math.log(1 / dprime)) + T * epsAmp * (Math.exp(epsAmp) - 1);
    var snr = signal / (noiseStd + 1e-12);
    say('single release: ε = <b>' + DL.fmt(eps1, 2) + '</b> at δ = ' + d.toExponential(0) + ' &nbsp; | &nbsp; DP-SGD: q = ' + DL.fmt(q, 4) + ', T = ' + T + ' &nbsp; rough total ε ≲ <b>' + DL.fmt(Math.min(epsTotal, T * epsAmp), 1) + '</b> (heuristic bound, loose) &nbsp; | &nbsp; per-coordinate noise std on the averaged gradient = ' + noiseStd.toExponential(2) + ', signal/noise for a clipped unit gradient ≈ <b>' + DL.fmt(snr, 1) + '</b>');
  }
  c.draw = draw;
  draw();
});
