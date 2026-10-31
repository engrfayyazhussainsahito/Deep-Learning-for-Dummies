DL.register('week04-bias-variance', function (root) {
  var ctrl = DL.controls(root);
  var deg = DL.slider(ctrl, { label: 'Polynomial degree', min: 0, max: 15, step: 1, value: 3 }, draw);
  var n = DL.slider(ctrl, { label: 'Training points n', min: 8, max: 80, step: 1, value: 20 }, draw);
  var sig = DL.slider(ctrl, { label: 'Noise σ', min: 0.0, max: 0.8, step: 0.05, value: 0.3 }, draw);
  var lam = DL.slider(ctrl, { label: 'log10 ridge λ', min: -8, max: 1, step: 0.5, value: -8, format: function (v) { return v <= -8 ? 'off' : '1e' + v; } }, draw);
  var row = DL.row(root);
  var seedBtn = DL.button(row, 'Resample training set', function () { seed += 1; draw(); });
  var showMany = DL.toggle(row, 'Show 30 other fits', true, draw);
  var seed = 1;
  var c = DL.canvas(root, { aspect: 0.55 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Truth: sin(πx) on [-1, 1]. Faint curves are fits to independent training sets of the same size; their spread is the variance, their mean offset from the truth is the bias. Try degree 1, 3, 12 with n = 20, then raise λ at degree 12.' }));
  function truth(x) { return Math.sin(Math.PI * x); }
  // Chebyshev basis for conditioning
  function basis(x, d) { var b = [1, x]; for (var k = 2; k <= d; k++) b.push(2 * x * b[k - 1] - b[k - 2]); return b.slice(0, d + 1); }
  function fit(xs, ys, d, l) {
    var m = d + 1, A = [], r = [], i, j, k;
    for (i = 0; i < m; i++) { A.push(new Array(m).fill(0)); r.push(0); }
    for (k = 0; k < xs.length; k++) {
      var b = basis(xs[k], d);
      for (i = 0; i < m; i++) { r[i] += b[i] * ys[k]; for (j = 0; j < m; j++) A[i][j] += b[i] * b[j]; }
    }
    for (i = 0; i < m; i++) A[i][i] += l + 1e-10;
    for (i = 0; i < m; i++) { // gaussian elimination with pivoting
      var p = i;
      for (j = i + 1; j < m; j++) if (Math.abs(A[j][i]) > Math.abs(A[p][i])) p = j;
      var t = A[i]; A[i] = A[p]; A[p] = t; var tr = r[i]; r[i] = r[p]; r[p] = tr;
      for (j = i + 1; j < m; j++) { var f = A[j][i] / A[i][i]; for (k = i; k < m; k++) A[j][k] -= f * A[i][k]; r[j] -= f * r[i]; }
    }
    var w = new Array(m).fill(0);
    for (i = m - 1; i >= 0; i--) { var s = r[i]; for (j = i + 1; j < m; j++) s -= A[i][j] * w[j]; w[i] = s / A[i][i]; }
    return w;
  }
  function pred(w, x) { var b = basis(x, w.length - 1), s = 0; for (var i = 0; i < w.length; i++) s += w[i] * b[i]; return s; }
  function sample(N, s, sd) { var rnd = DL.rng(sd * 7919 + 13), xs = [], ys = []; for (var i = 0; i < N; i++) { var x = -1 + 2 * (i + rnd()) / N; xs.push(x); ys.push(truth(x) + s * DL.randn(rnd)); } return { xs: xs, ys: ys }; }
  function draw() {
    var d = Math.round(deg.get()), N = Math.round(n.get()), s = sig.get(), l = lam.get() <= -8 ? 0 : Math.pow(10, lam.get());
    var col = DL.colors();
    var P = DL.plot(c, { xmin: -1, xmax: 1, ymin: -2, ymax: 2, xlabel: 'x', ylabel: 'y' });
    P.clear(); P.grid(); P.axes();
    var grid = DL.linspace(-1, 1, 101), K = 30, preds = [], k, i;
    for (k = 0; k < K; k++) { var D = sample(N, s, 1000 + k); var w = fit(D.xs, D.ys, d, l); var ys = grid.map(function (x) { return pred(w, x); }); preds.push(ys); if (showMany.get()) P.line(grid, ys.map(function (v) { return DL.clamp(v, -3, 3); }), col.muted + '55', 1); }
    var mean = grid.map(function (_, i) { var s2 = 0; for (var k = 0; k < K; k++) s2 += preds[k][i]; return s2 / K; });
    var bias2 = 0, vr = 0;
    for (i = 0; i < grid.length; i++) { var t = truth(grid[i]); bias2 += Math.pow(mean[i] - t, 2); for (k = 0; k < K; k++) vr += Math.pow(preds[k][i] - mean[i], 2) / K; }
    bias2 /= grid.length; vr /= grid.length;
    P.fn(truth, col.green, 2.5);
    var D0 = sample(N, s, seed), w0 = fit(D0.xs, D0.ys, d, l);
    P.line(grid, grid.map(function (x) { return DL.clamp(pred(w0, x), -3, 3); }), col.red, 2.8);
    P.points(D0.xs, D0.ys, col.blue, 3.5);
    var tr = 0; for (i = 0; i < N; i++) tr += Math.pow(pred(w0, D0.xs[i]) - D0.ys[i], 2); tr /= N;
    var te = 0, T = sample(400, s, 99991); for (i = 0; i < 400; i++) te += Math.pow(pred(w0, T.xs[i]) - T.ys[i], 2); te /= 400;
    say('degree <b>' + d + '</b> &nbsp; train MSE <b>' + DL.fmt(tr, 3) + '</b> &nbsp; test MSE <b>' + DL.fmt(te, 3) + '</b><br>over 30 resamples: bias² <b>' + DL.fmt(bias2, 3) + '</b> &nbsp; variance <b>' + DL.fmt(vr, 3) + '</b> &nbsp; noise σ² <b>' + DL.fmt(s * s, 3) + '</b> &nbsp; sum ≈ <b>' + DL.fmt(bias2 + vr + s * s, 3) + '</b>');
  }
  c.draw = draw;
});
