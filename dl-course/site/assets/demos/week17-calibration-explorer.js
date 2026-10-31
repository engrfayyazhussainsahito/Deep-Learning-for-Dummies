DL.register('week17-calibration-explorer', function (root) {
  // Synthetic 5-class classifier: base logits are calibrated by construction (labels are sampled from softmax(base)).
  // The model outputs base * s / T. s>1 -> overconfident, s<1 -> underconfident. Temperature T rescales back.
  var ctrl = DL.controls(root);
  var infl = DL.slider(ctrl, { label: 'Logit inflation s', min: 0.4, max: 4, step: 0.05, value: 2.5, format: function (v) { return DL.fmt(v, 2); } }, draw);
  var Tt = DL.slider(ctrl, { label: 'Temperature T', min: 0.3, max: 6, step: 0.05, value: 1, format: function (v) { return DL.fmt(v, 2); } }, draw);
  var nb = DL.slider(ctrl, { label: 'Bins', min: 5, max: 30, step: 1, value: 10, format: function (v) { return String(Math.round(v)); } }, draw);
  var row = DL.row(root);
  DL.button(row, 'Fit T (min NLL)', function () { Tt.set(fitT()); draw(); });
  DL.button(row, 'Resample', function () { seed++; gen(); draw(); });
  var c = DL.canvas(root, { aspect: 0.62 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Bars show accuracy per confidence bin; the dashed diagonal is perfect calibration; red gaps are overconfidence. Set the inflation above 1 and watch the bars sink below the diagonal and ECE grow. Click "Fit T" to find the temperature that minimizes NLL: it is close to the inflation, accuracy never changes, and the diagram snaps back to the diagonal. Try inflation < 1 (underconfidence) to see T < 1.' }));
  var seed = 3, K = 5, N = 3000, base = [], lab = [];
  function gen() {
    var r = DL.rng(seed), g = function () { return DL.randn(r); };
    base = []; lab = [];
    for (var i = 0; i < N; i++) {
      var z = [], k;
      for (k = 0; k < K; k++) z.push(g() * 1.3);
      var p = DL.softmax(z), u = r(), a = 0, y = K - 1;
      for (k = 0; k < K; k++) { a += p[k]; if (u <= a) { y = k; break; } }
      base.push(z); lab.push(y);
    }
  }
  function probs(s, T) { return base.map(function (z) { return DL.softmax(z.map(function (v) { return v * s / T; })); }); }
  function nll(T) {
    var P = probs(infl.get(), T), a = 0;
    for (var i = 0; i < N; i++) a -= Math.log(Math.max(P[i][lab[i]], 1e-12));
    return a / N;
  }
  function fitT() {
    var lo = 0.3, hi = 6, i;
    for (i = 0; i < 40; i++) {   // ternary search: NLL is convex in 1/T, unimodal in T
      var m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3;
      if (nll(m1) < nll(m2)) hi = m2; else lo = m1;
    }
    return (lo + hi) / 2;
  }
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), ctx = c.ctx, W = c.w, H = c.h, B = Math.round(nb.get());
    var T = Tt.get(), P = probs(infl.get(), T);
    var cnt = [], conf = [], acc = [], i, b;
    for (b = 0; b < B; b++) { cnt.push(0); conf.push(0); acc.push(0); }
    var correct = 0, nl = 0, brier = 0;
    for (i = 0; i < N; i++) {
      var p = P[i], mx = 0, am = 0, k;
      for (k = 0; k < K; k++) if (p[k] > mx) { mx = p[k]; am = k; }
      b = Math.min(B - 1, Math.floor(mx * B));
      cnt[b]++; conf[b] += mx; if (am === lab[i]) { acc[b]++; correct++; }
      nl -= Math.log(Math.max(p[lab[i]], 1e-12));
      for (k = 0; k < K; k++) brier += Math.pow(p[k] - (k === lab[i] ? 1 : 0), 2);
    }
    var ece = 0;
    var pl = 46, pr = 12, pt = 12, pb = 36, pw = W - pl - pr, ph = H - pt - pb;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = col.line; ctx.fillStyle = col.muted; ctx.font = '11px sans-serif'; ctx.lineWidth = 1;
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (var t = 0; t <= 5; t++) { var yy = pt + ph - t / 5 * ph; ctx.beginPath(); ctx.moveTo(pl, yy); ctx.lineTo(pl + pw, yy); ctx.stroke(); ctx.fillText(DL.fmt(t / 5, 1), pl - 6, yy); }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (t = 0; t <= 5; t++) ctx.fillText(DL.fmt(t / 5, 1), pl + t / 5 * pw, pt + ph + 5);
    ctx.fillText('confidence', pl + pw / 2, H - 14);
    ctx.save(); ctx.translate(12, pt + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'top'; ctx.fillText('accuracy', 0, 0); ctx.restore();
    var bw = pw / B;
    for (b = 0; b < B; b++) {
      if (!cnt[b]) continue;
      var a = acc[b] / cnt[b], cf = conf[b] / cnt[b];
      ece += cnt[b] / N * Math.abs(a - cf);
      var x = pl + b * bw;
      ctx.fillStyle = col.accent; ctx.globalAlpha = 0.8; ctx.fillRect(x + 1, pt + ph - a * ph, bw - 2, a * ph); ctx.globalAlpha = 1;
      // gap between accuracy and mean confidence
      ctx.fillStyle = a < cf ? col.red : col.green; ctx.globalAlpha = 0.45;
      var y1 = pt + ph - Math.max(a, cf) * ph, y2 = pt + ph - Math.min(a, cf) * ph;
      ctx.fillRect(x + 1, y1, bw - 2, Math.max(1, y2 - y1)); ctx.globalAlpha = 1;
    }
    ctx.setLineDash([5, 4]); ctx.strokeStyle = col.ink; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(pl, pt + ph); ctx.lineTo(pl + pw, pt); ctx.stroke(); ctx.setLineDash([]);
    say('accuracy <b>' + DL.fmt(100 * correct / N, 1) + '%</b> (unchanged by T) &nbsp; ECE <b>' + DL.fmt(ece, 4) + '</b> &nbsp; NLL <b>' + DL.fmt(nl / N, 3) + '</b> &nbsp; Brier <b>' + DL.fmt(brier / N, 3) + '</b> &nbsp; red = overconfident bins, green = underconfident. The perfectly calibrated setting is s = 1, T = 1 (or T = s).');
  }
  c.draw = draw;
  gen(); draw();
});
