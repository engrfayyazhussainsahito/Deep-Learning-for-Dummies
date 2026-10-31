DL.register('week03-init-explorer', function (root) {
  var ctrl = DL.controls(root);
  var depth = DL.slider(ctrl, { label: 'Depth (layers)', min: 2, max: 40, step: 1, value: 20 }, run);
  var width = DL.slider(ctrl, { label: 'Width', min: 16, max: 256, step: 16, value: 128 }, run);
  var scale = DL.slider(ctrl, { label: 'Weight std / He std (log2)', min: -3, max: 3, step: 0.1, value: 0, format: function (v) { return 'x' + Math.pow(2, v).toFixed(2); } }, run);
  var act = DL.select(ctrl, { label: 'Activation', value: 'relu', options: [{ value: 'relu', label: 'ReLU' }, { value: 'tanh', label: 'tanh' }, { value: 'lin', label: 'linear' }] }, run);
  var row = DL.row(root);
  var rule = DL.select(row, { label: 'Base rule', value: 'he', options: [{ value: 'he', label: 'He: 2/n_in' }, { value: 'xavier', label: 'Glorot: 2/(n_in+n_out)' }, { value: 'lecun', label: 'LeCun: 1/n_in' }] }, run);
  DL.button(row, 'New random draw', function () { seed++; run(); });
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Solid line: standard deviation of the activations in each layer (forward pass, unit-variance inputs). Dashed line: standard deviation of the backpropagated gradient with respect to each layer input. With the right scale (x1.00 and the matching rule) both stay flat; a factor off by 2 compounds exponentially with depth. The vertical axis is logarithmic.' }));
  var seed = 1;
  var data = null;
  function run() {
    var L = depth.get(), n = width.get(), r = DL.rng(seed * 7919), B = 64;
    var base = rule.get() === 'he' ? 2 / n : rule.get() === 'xavier' ? 2 / (n + n) : 1 / n;
    var sd = Math.sqrt(base) * Math.pow(2, scale.get()), a = act.get();
    var phi = function (z) { return a === 'relu' ? Math.max(0, z) : a === 'tanh' ? Math.tanh(z) : z; };
    var dphi = function (z, h) { return a === 'relu' ? (z > 0 ? 1 : 0) : a === 'tanh' ? 1 - h * h : 1; };
    var X = [], i, j, k, l;
    for (i = 0; i < B; i++) { X[i] = []; for (j = 0; j < n; j++) X[i][j] = DL.randn(r); }
    var Ws = [], Zs = [], Hs = [X], fwd = [rms(X)];
    for (l = 0; l < L; l++) {
      var W = []; for (j = 0; j < n; j++) { W[j] = []; for (k = 0; k < n; k++) W[j][k] = sd * DL.randn(r); }
      var Z = DL.matmul(Hs[l], W), H = Z.map(function (row) { return row.map(phi); });
      Ws.push(W); Zs.push(Z); Hs.push(H); fwd.push(rms(H));
    }
    // backward: random unit-variance gradient at the output
    var G = []; for (i = 0; i < B; i++) { G[i] = []; for (j = 0; j < n; j++) G[i][j] = DL.randn(r); }
    var bwd = [rms(G)];
    for (l = L - 1; l >= 0; l--) {
      var dZ = G.map(function (row, ii) { return row.map(function (g, jj) { return g * dphi(Zs[l][ii][jj], Hs[l + 1][ii][jj]); }); });
      G = DL.matmul(dZ, DL.transpose(Ws[l]));
      bwd.unshift(rms(G));
    }
    data = { fwd: fwd, bwd: bwd, L: L, sd: sd };
    draw();
  }
  function rms(M) { var s = 0, cnt = 0; for (var i = 0; i < M.length; i++) for (var j = 0; j < M[i].length; j++) { s += M[i][j] * M[i][j]; cnt++; } return Math.sqrt(s / cnt); }
  function draw() {
    if (!data) return;
    var col = DL.colors(), L = data.L;
    var lo = -8, hi = 8;
    var P = DL.plot(c, { xmin: 0, xmax: L, ymin: lo, ymax: hi, xlabel: 'layer', ylabel: 'log10 std' });
    P.clear(); P.grid(); P.axes();
    var lg = function (v) { return Math.max(lo, Math.min(hi, Math.log10(Math.max(v, 1e-30)))); };
    P.hline(0, col.muted);
    P.line(DL.range(L + 1), data.fwd.map(lg), col.blue, 2.5);
    P.line(DL.range(L + 1), data.bwd.map(lg), col.red, 2.2, [6, 4]);
    P.text('forward activations', L * 0.05, hi - 1, col.blue);
    P.text('backward gradients (plotted by layer input)', L * 0.05, hi - 2, col.red);
    var fL = data.fwd[L], b0 = data.bwd[0];
    say('weight std = <b>' + DL.fmt(data.sd, 4) + '</b> &nbsp; activation std at last layer = <b>' + (fL < 1e-6 || fL > 1e6 ? fL.toExponential(1) : DL.fmt(fL, 3)) + '</b> &nbsp; gradient std at first layer = <b>' + (b0 < 1e-6 || b0 > 1e6 ? b0.toExponential(1) : DL.fmt(b0, 3)) + '</b> (output gradient std = 1)');
  }
  c.draw = draw;
  run();
});
