DL.register('week09-attention-explorer', function (root) {
  var D = 64; // embedding width
  var ctrl = DL.controls(root);
  var row0 = DL.row(ctrl);
  var inp = DL.el('input', { type: 'text', value: 'the cat sat on the mat because it was tired', class: 'dl-text', style: { width: '100%', maxWidth: '520px' }, 'aria-label': 'Type tokens separated by spaces' });
  row0.appendChild(DL.el('span', { class: 'dl-slider-label', text: 'Tokens ' }));
  row0.appendChild(inp);
  var dk = DL.slider(ctrl, { label: 'Head width d_k', min: 1, max: 256, step: 1, value: 64 }, draw);
  var temp = DL.slider(ctrl, { label: 'Temperature τ', min: 0.1, max: 4, step: 0.05, value: 1 }, draw);
  var scale = DL.toggle(ctrl, 'Scale by 1/√d_k', true, draw);
  var causal = DL.toggle(ctrl, 'Causal mask', false, draw);
  var pos = DL.toggle(ctrl, 'Add sinusoidal positions', false, draw);
  var seed = DL.slider(ctrl, { label: 'Weight seed', min: 1, max: 20, step: 1, value: 3 }, draw);
  var c = DL.canvas(root, { aspect: 0.7 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Type any words. Each token gets a fixed random embedding (identical words share one), then Q = XW_Q and K = XW_K with random weights: no training, so attention reflects only geometry. Try: turn scaling off and raise d_k to 256 and watch rows collapse to one-hot (softmax saturation); lower τ for the same effect; turn on the causal mask and see the upper triangle vanish; repeat a word to see equal columns. Click a row to inspect it.' }));
  var sel = -1;
  function hash(s) { var h = 2166136261; for (var i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
  function tokens() {
    var t = inp.value.toLowerCase().split(/\s+/).filter(function (s) { return s.length; });
    return t.slice(0, 14);
  }
  function embed(tok, p) {
    var r = DL.rng(hash(tok)), v = [], i;
    for (i = 0; i < D; i++) v.push(DL.randn(r));
    if (pos.get()) for (i = 0; i < D; i += 2) { var w = Math.pow(10000, -i / D); v[i] += Math.sin(p * w) * 1.5; v[i + 1] += Math.cos(p * w) * 1.5; }
    return v;
  }
  function mat(rows, cols, sd, s) { var r = DL.rng(s), M = [], i, j; for (i = 0; i < rows; i++) { M[i] = []; for (j = 0; j < cols; j++) M[i][j] = DL.randn(r) * sd; } return M; }
  function compute() {
    var tk = tokens(), n = tk.length;
    if (!n) return null;
    var X = tk.map(function (t, i) { return embed(t, i); });
    var d = Math.round(dk.get()), s0 = Math.round(seed.get()) * 101;
    var Wq = mat(D, d, 1 / Math.sqrt(D), s0 + 1), Wk = mat(D, d, 1 / Math.sqrt(D), s0 + 2);
    var Q = DL.matmul(X, Wq), K = DL.matmul(X, Wk), S = DL.matmul(Q, DL.transpose(K)), i, j;
    var f = scale.get() ? 1 / Math.sqrt(d) : 1, A = [], raw = [];
    for (i = 0; i < n; i++) {
      var z = [];
      for (j = 0; j < n; j++) { var v = S[i][j] * f; z.push(causal.get() && j > i ? -Infinity : v); if (!(causal.get() && j > i)) raw.push(v); }
      var m = Math.max.apply(null, z), e = z.map(function (u) { return u === -Infinity ? 0 : Math.exp((u - m) / temp.get()); }), sm = DL.sum(e);
      A.push(e.map(function (u) { return u / sm; }));
    }
    return { tk: tk, A: A, raw: raw, n: n, d: d };
  }
  function draw() {
    var ctx = c.ctx, col = DL.colors(), R = compute();
    ctx.clearRect(0, 0, c.w, c.h);
    if (!R) { say('Type at least one token.'); return; }
    var n = R.n, x0 = 70, y0 = 62, side = Math.min(c.w - x0 - 16, c.h - y0 - 10);
    DL.heatmap(ctx, R.A, x0, y0, side, side, { min: 0, max: 1, cmap: DL.cmap.viridis, labels: n <= 10, rowLabels: R.tk, colLabels: R.tk, fmt: function (v) { return v < 0.005 ? '' : v.toFixed(2); } });
    ctx.fillStyle = col.muted; ctx.font = '11px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('keys j →', x0, 12); ctx.fillText('queries i ↓', 4, y0 - 26);
    if (sel >= 0 && sel < n) { ctx.strokeStyle = col.accent; ctx.lineWidth = 2.5; ctx.strokeRect(x0, y0 + sel * side / n, side, side / n); }
    c._g = { x0: x0, y0: y0, side: side, n: n };
    var mean = DL.sum(R.raw) / R.raw.length, sd = Math.sqrt(DL.sum(R.raw.map(function (v) { return (v - mean) * (v - mean); })) / R.raw.length);
    var ent = R.A.map(function (r) { var s = 0; r.forEach(function (p) { if (p > 0) s -= p * Math.log(p); }); return s; });
    var maxp = R.A.map(function (r) { return Math.max.apply(null, r); });
    var t = 'd_k = <b>' + R.d + '</b>; std of logits (after optional scaling) = <b>' + DL.fmt(sd, 2) + '</b> (≈ 1 when scaled, ≈ √d_k = ' + DL.fmt(Math.sqrt(R.d), 1) + ' when not)<br>' +
      'mean row entropy = <b>' + DL.fmt(DL.sum(ent) / n, 2) + '</b> nats; mean max weight = <b>' + DL.fmt(DL.sum(maxp) / n, 2) + '</b>';
    if (sel >= 0 && sel < n) {
      var top = R.A[sel].map(function (p, j) { return [p, j]; }).sort(function (a, b) { return b[0] - a[0]; }).slice(0, 3);
      t += '<br>Query "' + R.tk[sel] + '" output = ' + top.map(function (u) { return DL.fmt(u[0], 2) + '·v(' + R.tk[u[1]] + ')'; }).join(' + ') + ' + ...';
    } else t += '<br>Click a row to see its output mixture.';
    say(t);
  }
  c.canvas.addEventListener('click', function (ev) {
    var r = c.canvas.getBoundingClientRect(), y = ev.clientY - r.top, g = c._g;
    if (!g) return;
    var k = Math.floor((y - g.y0) / (g.side / g.n));
    sel = k >= 0 && k < g.n ? k : -1; draw();
  });
  inp.addEventListener('input', function () { sel = -1; draw(); });
  c.draw = draw;
  draw();
});
