DL.register('week07-vanishing-gradient', function (root) {
  var N = 24;
  var ctrl = DL.controls(root);
  var rho = DL.slider(ctrl, { label: 'Spectral radius ρ of W', min: 0.3, max: 2.0, step: 0.01, value: 0.9 }, draw);
  var T = DL.slider(ctrl, { label: 'Sequence length T', min: 10, max: 100, step: 1, value: 50 }, draw);
  var inp = DL.slider(ctrl, { label: 'Input scale', min: 0, max: 3, step: 0.05, value: 0.5 }, draw);
  var act = DL.select(ctrl, { label: 'Activation', value: 'tanh', options: [{ value: 'tanh', label: 'tanh' }, { value: 'identity', label: 'identity (linear RNN)' }, { value: 'relu', label: 'ReLU' }] }, draw);
  var row = DL.row(root);
  DL.button(row, 'New random draw', function () { seed++; draw(); });
  var seed = 1;
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Plots log10 of the norm of the gradient that reaches a state k steps in the past, relative to the gradient at the last step. The dashed line is the linear prediction k·log10(ρ). Try ρ = 0.9 (vanishing), ρ = 1.0 and ρ = 1.5 with the identity activation (exploding), then switch to tanh and see how saturation (large input scale) pulls the curve down even when ρ > 1.' }));

  function compute() {
    var rand = DL.rng(seed * 7919), rn = function () { return DL.randn(rand); }, n = N, r = rho.get(), t = Math.round(T.get());
    var W = [], U = [], i, j;
    for (i = 0; i < n; i++) { W.push([]); U.push([]); for (j = 0; j < n; j++) { W[i].push(rn() * r / Math.sqrt(n)); U[i].push(rn() / Math.sqrt(n)); } }
    var h = [], d = [], kind = act.get(), s = inp.get(), step;
    var hp = []; for (i = 0; i < n; i++) hp.push(0);
    for (step = 0; step < t; step++) {
      var a = [], dd = [];
      for (i = 0; i < n; i++) {
        var z = 0; for (j = 0; j < n; j++) z += W[i][j] * hp[j] + U[i][j] * s * rn();
        if (kind === 'tanh') { a.push(Math.tanh(z)); dd.push(1 - Math.tanh(z) * Math.tanh(z)); }
        else if (kind === 'relu') { a.push(Math.max(0, z)); dd.push(z > 0 ? 1 : 0); }
        else { a.push(z); dd.push(1); }
        if (!isFinite(a[i])) a[i] = 0;
        a[i] = Math.max(-1e6, Math.min(1e6, a[i]));
      }
      hp = a; d.push(dd);
    }
    // backward: g_T = random unit vector; g_{t-1} = W^T (d_t * g_t)
    var g = []; var nn = 0;
    for (i = 0; i < n; i++) { g.push(rn()); nn += g[i] * g[i]; }
    nn = Math.sqrt(nn); g = g.map(function (v) { return v / nn; });
    var out = [0];
    for (step = t - 1; step >= 1; step--) {
      var u = g.map(function (v, k) { return v * d[step][k]; }), ng = [], m = 0;
      for (j = 0; j < n; j++) { var q = 0; for (i = 0; i < n; i++) q += W[i][j] * u[i]; ng.push(q); m += q * q; }
      g = ng; out.push(Math.log10(Math.sqrt(m) + 1e-300));
    }
    return out;
  }
  function draw() {
    var ys = compute(), col = DL.colors(), r = rho.get(), xs = ys.map(function (_, k) { return k; });
    var lo = -12, hi = 12;
    var P = DL.plot(c, { xmin: 0, xmax: Math.max(10, xs.length - 1), ymin: lo, ymax: hi, xlabel: 'steps back in time k', ylabel: 'log10 gradient norm' });
    P.clear(); P.grid(); P.axes();
    P.hline(0, col.muted);
    var th = xs.map(function (k) { return Math.max(lo, Math.min(hi, k * Math.log10(r))); });
    P.line(xs, th, col.amber, 1.8, [6, 4]);
    P.line(xs, ys.map(function (v) { return Math.max(lo, Math.min(hi, v)); }), col.accent, 2.6);
    var last = ys[ys.length - 1];
    var verdict = last < -3 ? 'vanishing: early inputs get essentially no learning signal' : (last > 3 ? 'exploding: updates will be dominated by long-range terms (clip!)' : 'roughly stable over this horizon');
    say('ρ = <b>' + DL.fmt(r, 2) + '</b> &nbsp; gradient at k = ' + (ys.length - 1) + ': 10<sup>' + DL.fmt(last, 1) + '</sup> &nbsp; linear prediction: 10<sup>' + DL.fmt((ys.length - 1) * Math.log10(r), 1) + '</sup> &nbsp; → ' + verdict);
  }
  c.draw = draw;
  draw();
});
