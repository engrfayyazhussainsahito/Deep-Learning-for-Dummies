DL.register('week07-lstm-gates', function (root) {
  var H = 4;
  var ctrl = DL.controls(root);
  var bf = DL.slider(ctrl, { label: 'Forget-gate bias', min: -3, max: 5, step: 0.25, value: 1 }, reset);
  var seqSel = DL.select(ctrl, { label: 'Input sequence', value: 'pulse', options: [{ value: 'pulse', label: 'one pulse, then silence' }, { value: 'sine', label: 'slow sine wave' }, { value: 'pulses', label: 'two pulses (opposite sign)' }] }, reset);
  var row = DL.row(root);
  var t = 0, X = [], hist = [], st = null, timer = null;
  var W; // gate weights: for each of f,i,g,o : H x (H+1) recurrent+input
  function build() {
    var rand = DL.rng(11), rn = function () { return DL.randn(rand); };
    W = {};
    ['f', 'i', 'g', 'o'].forEach(function (k) {
      W[k] = []; for (var u = 0; u < H; u++) { var r = []; for (var j = 0; j < H; j++) r.push(rn() * 0.6); r.push(rn() * 1.5 + (k === 'i' || k === 'g' ? 1.5 : 0)); W[k].push(r); }
    });
  }
  build();
  function seq() {
    var s = seqSel.get(), a = [], i;
    for (i = 0; i < 40; i++) a.push(s === 'pulse' ? (i >= 4 && i < 7 ? 2 : 0) : s === 'sine' ? 1.5 * Math.sin(i / 4) : (i >= 3 && i < 6 ? 2 : (i >= 20 && i < 23 ? -2 : 0)));
    return a;
  }
  function sig(x) { return 1 / (1 + Math.exp(-x)); }
  function stepFn(h, cc, x) {
    var o = { f: [], i: [], g: [], o: [], c: [], h: [] }, u, j;
    for (u = 0; u < H; u++) {
      var pre = {};
      ['f', 'i', 'g', 'o'].forEach(function (k) { var z = W[k][u][H] * x; for (j = 0; j < H; j++) z += W[k][u][j] * h[j]; pre[k] = z; });
      var f = sig(pre.f + bf.get()), i = sig(pre.i), g = Math.tanh(pre.g), oo = sig(pre.o);
      var c = f * cc[u] + i * g;
      o.f.push(f); o.i.push(i); o.g.push(g); o.o.push(oo); o.c.push(c); o.h.push(oo * Math.tanh(c));
    }
    return o;
  }
  function reset() { t = 0; X = seq(); hist = []; st = { h: [0, 0, 0, 0], c: [0, 0, 0, 0] }; draw(); }
  function advance() {
    if (t >= X.length) { stop(); return; }
    var o = stepFn(st.h, st.c, X[t]); o.x = X[t]; hist.push(o); st = { h: o.h, c: o.c }; t++; draw();
  }
  function stop() { if (timer) { clearInterval(timer); timer = null; } }
  DL.button(row, 'Step', function () { stop(); advance(); });
  DL.button(row, 'Play / pause', function () { if (timer) stop(); else timer = setInterval(advance, 350); });
  DL.button(row, 'Reset', function () { stop(); reset(); });
  var c = DL.canvas(root, { aspect: 0.62 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'A 4-unit LSTM with fixed random weights reads a scalar input x_t. Top: gate activations at the current step (rows) for each hidden unit (columns). Bottom: the input, and the four cell states over time. Raise the forget bias and watch the cell states hold their values after the pulse ends; lower it and they decay back toward zero. Each gate value is in (0,1) except g, which is in (-1,1).' }));
  function draw() {
    var ctx = c.ctx, col = DL.colors(), w = c.w, h = c.h;
    ctx.clearRect(0, 0, w, h);
    var cur = hist.length ? hist[hist.length - 1] : null;
    var rows = ['f', 'i', 'g', 'o', 'c', 'h'], names = ['forget f', 'input i', 'candidate g', 'output o', 'cell c', 'hidden h'];
    var M = rows.map(function (k) { return cur ? cur[k] : [0, 0, 0, 0]; });
    var gx = 90, gy = 22, gw = Math.min(w * 0.42, 260), gh = h * 0.42;
    DL.heatmap(ctx, M, gx, gy, gw, gh, { min: -1, max: 1, cmap: DL.cmap.diverge, labels: true, rowLabels: names, colLabels: ['u1', 'u2', 'u3', 'u4'], fmt: function (v) { return v.toFixed(2); } });
    ctx.fillStyle = col.muted; ctx.font = '12px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    var tx = gx + gw + 20;
    ctx.fillText('step t = ' + t + ',  x_t = ' + (cur ? cur.x.toFixed(2) : '-'), tx, gy);
    ctx.fillText('c_t = f ⊙ c_(t-1) + i ⊙ g', tx, gy + 20);
    ctx.fillText('h_t = o ⊙ tanh(c_t)', tx, gy + 38);
    ctx.fillText('mean forget gate = ' + (cur ? DL.fmt(DL.sum(cur.f) / H, 2) : '-'), tx, gy + 62);
    // time series
    var P = DL.plot(c, { xmin: 0, xmax: 40, ymin: -2.5, ymax: 2.5, xlabel: 'time step', ylabel: 'x and c', pad: { l: 44, r: 14, t: gy + gh + 16, b: 34 } });
    P.grid(); P.axes();
    var xs = hist.map(function (_, k) { return k; });
    P.line(X.map(function (_, k) { return k; }), X, col.muted, 1.2, [4, 3]);
    var cols = [col.blue, col.green, col.purple, col.amber];
    for (var u = 0; u < H; u++) { if (hist.length) P.line(xs, hist.map(function (o) { return Math.max(-2.5, Math.min(2.5, o.c[u])); }), cols[u], 2); }
    if (t > 0) P.vline(t - 1, col.accent);
    say('dashed grey = input x_t; coloured = cell states c_t of the four units. Forget bias = <b>' + DL.fmt(bf.get(), 2) + '</b> (σ = ' + DL.fmt(sig(bf.get()), 2) + ' when the rest of the pre-activation is zero).');
  }
  c.draw = draw;
  reset();
});
