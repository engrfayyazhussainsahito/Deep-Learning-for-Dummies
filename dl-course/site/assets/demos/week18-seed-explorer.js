DL.register('week18-seed-explorer', function (root) {
  // Two models A and B with true mean accuracies and run-to-run (seed) standard deviation.
  // Draw n seeds each; show points, mean +- 95% t-interval, and a Welch t-test. Repeat many "papers" to
  // show how often a single seed picks the wrong winner.
  var ctrl = DL.controls(root);
  var gap = DL.slider(ctrl, { label: 'True gap B - A (points)', min: 0, max: 2, step: 0.05, value: 0.3, format: function (v) { return DL.fmt(v, 2); } }, function () { sweep(); draw(); });
  var sd = DL.slider(ctrl, { label: 'Seed std dev (points)', min: 0.05, max: 1.5, step: 0.05, value: 0.5, format: function (v) { return DL.fmt(v, 2); } }, function () { sweep(); draw(); });
  var nn = DL.slider(ctrl, { label: 'Seeds per model', min: 1, max: 20, step: 1, value: 3, format: function (v) { return String(Math.round(v)); } }, function () { sweep(); draw(); });
  var row = DL.row(root);
  DL.button(row, 'Rerun experiment', function () { seed += 7; gen(); draw(); });
  var c = DL.canvas(root, { aspect: 0.55 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Dots are final accuracies from independent seeds; bars are mean and 95% confidence interval (t distribution). With 1 seed you cannot even draw an interval. Click "Rerun experiment" repeatedly with a small gap: the apparent winner flips. The sweep line reports, over 400 simulated papers, how often B is declared better than A by the mean alone, and how often the Welch test (p < 0.05) agrees. Raise seeds to see the interval shrink like 1/sqrt(n).' }));
  var seed = 11, A = [], B = [], base = 80;
  var tcrit = [0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086];
  function tc(df) { df = Math.max(1, Math.round(df)); return df <= 20 ? tcrit[df] : (df <= 30 ? 2.05 : 1.97); }
  function stats(a) {
    var n = a.length, m = DL.sum(a) / n, v = 0;
    for (var i = 0; i < n; i++) v += (a[i] - m) * (a[i] - m);
    v = n > 1 ? v / (n - 1) : 0;
    return { n: n, m: m, v: v, ci: n > 1 ? tc(n - 1) * Math.sqrt(v / n) : 0 };
  }
  function draws(r, mu, n) { var o = []; for (var i = 0; i < n; i++) o.push(mu + sd.get() * DL.randn(r)); return o; }
  function gen() {
    var r = DL.rng(seed), n = Math.round(nn.get());
    A = draws(r, base, n); B = draws(r, base + gap.get(), n);
  }
  var sweepTxt = '';
  function welch(a, b) {
    var sa = stats(a), sb = stats(b), se = Math.sqrt(sa.v / sa.n + sb.v / sb.n);
    if (se === 0) return { t: 0, sig: false };
    var df = Math.pow(sa.v / sa.n + sb.v / sb.n, 2) / (Math.pow(sa.v / sa.n, 2) / Math.max(1, sa.n - 1) + Math.pow(sb.v / sb.n, 2) / Math.max(1, sb.n - 1));
    var t = (sb.m - sa.m) / se;
    return { t: t, sig: Math.abs(t) > tc(df) };
  }
  function sweep() {
    var r = DL.rng(999), n = Math.round(nn.get()), wins = 0, sigB = 0, sigWrong = 0, T = 400;
    for (var i = 0; i < T; i++) {
      var a = draws(r, base, n), b = draws(r, base + gap.get(), n);
      if (DL.sum(b) > DL.sum(a)) wins++;
      if (n > 1) { var w = welch(a, b); if (w.sig && w.t > 0) sigB++; if (w.sig && w.t < 0) sigWrong++; }
    }
    sweepTxt = 'Over ' + T + ' simulated papers: mean(B) &gt; mean(A) in <b>' + DL.fmt(wins / T * 100, 0) + '%</b>' +
      (n > 1 ? '; significant and correct in <b>' + DL.fmt(sigB / T * 100, 0) + '%</b>; significant and wrong in ' + DL.fmt(sigWrong / T * 100, 1) + '%.' : '; the Welch test needs at least 2 seeds.');
  }
  function draw() {
    if (!c.w) return;
    gen();
    var col = DL.colors(), ctx = c.ctx, W = c.w, H = c.h;
    ctx.clearRect(0, 0, W, H);
    var all = A.concat(B), lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    var sA = stats(A), sB = stats(B);
    lo = Math.min(lo, sA.m - sA.ci, sB.m - sB.ci, base - 1) - 0.2; hi = Math.max(hi, sA.m + sA.ci, sB.m + sB.ci, base + gap.get() + 1) + 0.2;
    var pl = 52, pr = 16, pt = 16, pb = 34, ph = H - pt - pb;
    function Y(v) { return pt + (hi - v) / (hi - lo) * ph; }
    ctx.strokeStyle = col.line; ctx.fillStyle = col.muted; ctx.font = '11px sans-serif'; ctx.lineWidth = 1; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (var t = 0; t <= 5; t++) { var v = lo + (hi - lo) * t / 5; ctx.beginPath(); ctx.moveTo(pl, Y(v)); ctx.lineTo(W - pr, Y(v)); ctx.stroke(); ctx.fillText(DL.fmt(v, 1), pl - 6, Y(v)); }
    ctx.save(); ctx.translate(12, pt + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText('test accuracy (%)', 0, 0); ctx.restore();
    var xs = [pl + (W - pl - pr) * 0.3, pl + (W - pl - pr) * 0.7], sets = [A, B], ss = [sA, sB], cols = [col.blue || col.muted, col.accent], lab = ['Model A (baseline)', 'Model B (new)'];
    ctx.setLineDash([4, 4]); ctx.strokeStyle = col.muted;
    [base, base + gap.get()].forEach(function (m) { ctx.beginPath(); ctx.moveTo(pl, Y(m)); ctx.lineTo(W - pr, Y(m)); ctx.stroke(); });
    ctx.setLineDash([]);
    var rr = DL.rng(seed + 1);
    for (var k = 0; k < 2; k++) {
      ctx.fillStyle = cols[k];
      sets[k].forEach(function (a) { ctx.globalAlpha = 0.75; ctx.beginPath(); ctx.arc(xs[k] + (rr() - 0.5) * 50, Y(a), 4.5, 0, 7); ctx.fill(); });
      ctx.globalAlpha = 1; ctx.strokeStyle = cols[k]; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(xs[k] - 38, Y(ss[k].m)); ctx.lineTo(xs[k] + 38, Y(ss[k].m)); ctx.stroke();
      if (ss[k].n > 1) {
        ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(xs[k] + 46, Y(ss[k].m - ss[k].ci)); ctx.lineTo(xs[k] + 46, Y(ss[k].m + ss[k].ci)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(xs[k] + 40, Y(ss[k].m - ss[k].ci)); ctx.lineTo(xs[k] + 52, Y(ss[k].m - ss[k].ci)); ctx.moveTo(xs[k] + 40, Y(ss[k].m + ss[k].ci)); ctx.lineTo(xs[k] + 52, Y(ss[k].m + ss[k].ci)); ctx.stroke();
      }
      ctx.fillStyle = col.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(lab[k], xs[k], H - pb + 8);
    }
    var w = A.length > 1 ? welch(A, B) : null;
    var verdict = A.length < 2 ? 'single seed: no uncertainty estimate' : (w.sig ? (w.t > 0 ? 'B significantly better (Welch, p&lt;0.05)' : 'A significantly better (Welch, p&lt;0.05)') : 'difference not significant');
    say('Observed gap: <b>' + DL.fmt(sB.m - sA.m, 2) + '</b> points (true ' + DL.fmt(gap.get(), 2) + '). ' + verdict + '.<br>' + sweepTxt);
  }
  c.draw = draw;
  sweep(); gen();
});
