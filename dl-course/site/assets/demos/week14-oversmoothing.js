DL.register('week14-oversmoothing', function (root) {
  // Two-community stochastic block model, 2-D node features, repeated GCN propagation (no weights, no nonlinearity).
  var n = 40, rand = DL.rng(11), comm = [], i, j;
  for (i = 0; i < n; i++) comm.push(i < n / 2 ? 0 : 1);
  var A = [];
  for (i = 0; i < n; i++) { A.push([]); for (j = 0; j < n; j++) A[i].push(0); }
  for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) { var p = comm[i] === comm[j] ? 0.30 : 0.04; if (rand() < p) A[i][j] = A[j][i] = 1; }
  var X0 = [];
  function gauss() { var u = Math.max(1e-9, rand()), v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.2832 * v); }
  for (i = 0; i < n; i++) X0.push([(comm[i] ? 1 : -1) * 1.0 + 1.1 * gauss(), 1.1 * gauss()]);
  // normalized adjacency with self loops
  var deg = [], S = [];
  for (i = 0; i < n; i++) { var d = 1; for (j = 0; j < n; j++) d += A[i][j]; deg.push(d); }
  for (i = 0; i < n; i++) { S.push([]); for (j = 0; j < n; j++) S[i].push(((A[i][j] || i === j) ? 1 : 0) / Math.sqrt(deg[i] * deg[j])); }

  var ctrl = DL.controls(root);
  var L = DL.slider(ctrl, { label: 'Layers (propagation steps)', min: 0, max: 64, step: 1, value: 0 }, draw);
  var al = DL.slider(ctrl, { label: 'Residual to input, alpha', min: 0, max: 0.6, step: 0.05, value: 0 }, draw);
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Left: node features after L steps of X <- (1-alpha) S X + alpha X0 (S = normalized adjacency with self-loops); colour = community. Right: normalized Dirichlet energy and centroid separation vs depth. With alpha = 0 the features collapse towards the dominant eigenvector of S (energy decays geometrically); alpha > 0 keeps a floor. Note: both communities collapse together, and class separation relative to spread is what a classifier needs.' }));

  function prop(X, a) {
    var Y = [];
    for (var r = 0; r < n; r++) {
      var s0 = 0, s1 = 0;
      for (var q = 0; q < n; q++) { var w = S[r][q]; if (w) { s0 += w * X[q][0]; s1 += w * X[q][1]; } }
      Y.push([(1 - a) * s0 + a * X0[r][0], (1 - a) * s1 + a * X0[r][1]]);
    }
    return Y;
  }
  function stats(X) {
    // normalized Dirichlet energy: sum over edges |x_i/sqrt(d_i) - x_j/sqrt(d_j)|^2 ; and separation = centroid distance / mean within-class std
    var e = 0, k, m0 = [0, 0], m1 = [0, 0], c0 = 0, c1 = 0;
    for (var a = 0; a < n; a++) for (var b = a + 1; b < n; b++) if (A[a][b]) {
      for (k = 0; k < 2; k++) { var dd = X[a][k] / Math.sqrt(deg[a]) - X[b][k] / Math.sqrt(deg[b]); e += dd * dd; }
    }
    for (a = 0; a < n; a++) { if (comm[a]) { m1[0] += X[a][0]; m1[1] += X[a][1]; c1++; } else { m0[0] += X[a][0]; m0[1] += X[a][1]; c0++; } }
    m0 = [m0[0] / c0, m0[1] / c0]; m1 = [m1[0] / c1, m1[1] / c1];
    var within = 0;
    for (a = 0; a < n; a++) { var m = comm[a] ? m1 : m0; within += Math.pow(X[a][0] - m[0], 2) + Math.pow(X[a][1] - m[1], 2); }
    within = Math.sqrt(within / n);
    var dist = Math.sqrt(Math.pow(m0[0] - m1[0], 2) + Math.pow(m0[1] - m1[1], 2));
    return { e: e, sep: dist / (within + 1e-12), dist: dist, within: within };
  }
  function trajectory(a) {
    var X = X0, out = [stats(X)], t;
    var Xs = [X];
    for (t = 1; t <= 64; t++) { X = prop(X, a); Xs.push(X); out.push(stats(X)); }
    return { stats: out, Xs: Xs };
  }
  function draw() {
    if (!c.w) return;
    var ctx = c.ctx, col = DL.colors(), W = c.w, H = c.h;
    var a = al.get(), l = Math.round(L.get());
    var T = trajectory(a), X = T.Xs[l], st = T.stats[l], e0 = T.stats[0].e;
    ctx.clearRect(0, 0, W, H);
    // left panel: scatter
    var pw = W * 0.46, ph = H - 40, ox = 10, oy = 24;
    ctx.strokeStyle = col.line; ctx.lineWidth = 1; ctx.strokeRect(ox, oy, pw, ph);
    ctx.fillStyle = col.muted; ctx.font = '12px sans-serif'; ctx.textAlign = 'left'; ctx.fillText('features after ' + l + ' layers (fixed axes)', ox, 16);
    var lim = 3.2;
    function sx(v) { return ox + (v + lim) / (2 * lim) * pw; }
    function sy(v) { return oy + ph - (v + lim) / (2 * lim) * ph; }
    for (var r = 0; r < n; r++) {
      ctx.beginPath(); ctx.arc(sx(Math.max(-lim, Math.min(lim, X[r][0]))), sy(Math.max(-lim, Math.min(lim, X[r][1]))), 4.2, 0, 6.2832);
      ctx.fillStyle = comm[r] ? col.amber : col.blue; ctx.globalAlpha = 0.85; ctx.fill(); ctx.globalAlpha = 1;
    }
    // right panel: curves
    var rx = W * 0.56, rw = W * 0.42, ry = 24, rh = ph;
    ctx.strokeStyle = col.line; ctx.strokeRect(rx, ry, rw, rh);
    ctx.fillStyle = col.muted; ctx.fillText('energy (blue, log) and separation (amber) vs depth', rx - 6, 16);
    function tx(t) { return rx + t / 64 * rw; }
    var lo = -6, hi = 0.3;
    function ey(v) { var q = Math.log10(Math.max(v / e0, 1e-6)); return ry + rh - (Math.max(lo, Math.min(hi, q)) - lo) / (hi - lo) * rh; }
    var smax = 0, t; for (t = 0; t <= 64; t++) smax = Math.max(smax, T.stats[t].sep);
    smax = Math.max(smax, 0.5);
    function sepy(v) { return ry + rh - Math.min(1, v / (smax * 1.1)) * rh; }
    ctx.lineWidth = 2.2; ctx.strokeStyle = col.blue; ctx.beginPath();
    for (t = 0; t <= 64; t++) { t ? ctx.lineTo(tx(t), ey(T.stats[t].e)) : ctx.moveTo(tx(t), ey(T.stats[t].e)); } ctx.stroke();
    ctx.strokeStyle = col.amber; ctx.beginPath();
    for (t = 0; t <= 64; t++) { t ? ctx.lineTo(tx(t), sepy(T.stats[t].sep)) : ctx.moveTo(tx(t), sepy(T.stats[t].sep)); } ctx.stroke();
    ctx.strokeStyle = col.ink; ctx.setLineDash([4, 4]); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(tx(l), ry); ctx.lineTo(tx(l), ry + rh); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = col.muted; ctx.textAlign = 'center'; ctx.fillText('layers 0 ... 64', rx + rw / 2, ry + rh + 14);
    say('L = <b>' + l + '</b>, alpha = <b>' + DL.fmt(a, 2) + '</b> &nbsp; energy / initial = <b>' + (st.e / e0 < 1e-4 ? (st.e / e0).toExponential(1) : DL.fmt(st.e / e0, 4)) + '</b> &nbsp; centroid distance = <b>' + DL.fmt(st.dist, 3) + '</b>, within-class spread = <b>' + DL.fmt(st.within, 3) + '</b>, separation ratio = <b>' + DL.fmt(st.sep, 2) + '</b>');
  }
  c.draw = draw;
});
