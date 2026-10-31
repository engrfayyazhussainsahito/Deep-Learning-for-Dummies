DL.register('week02-backprop-stepper', function (root) {
  var step = 0, MAXS = 8;
  var ctrl = DL.controls(root);
  function fmtS(v) { return (+v).toFixed(2); }
  var X = DL.slider(ctrl, { label: 'x', min: -3, max: 3, step: 0.1, value: 1.5, format: fmtS }, draw);
  var W = DL.slider(ctrl, { label: 'w', min: -3, max: 3, step: 0.1, value: 0.8, format: fmtS }, draw);
  var B = DL.slider(ctrl, { label: 'b', min: -3, max: 3, step: 0.1, value: -0.3, format: fmtS }, draw);
  var T = DL.slider(ctrl, { label: 'target t', min: 0, max: 1, step: 0.05, value: 1, format: fmtS }, draw);
  var row = DL.row(root);
  DL.button(row, '◀ Back', function () { step = Math.max(0, step - 1); draw(); });
  DL.button(row, 'Next step ▶', function () { step = Math.min(MAXS, step + 1); draw(); });
  DL.button(row, 'Reset', function () { step = 0; draw(); });
  DL.button(row, 'Run to end', function () { step = MAXS; draw(); });
  var c = DL.canvas(root, { aspect: 0.52 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Graph: p = x·w, z = p + b, a = σ(z), L = ½(a − t)². Steps 1 to 4 are the forward pass (values flow right); steps 5 to 8 are the backward pass (each node multiplies the gradient arriving from the right by its local derivative). Change a slider mid-way and watch every number update. The last line compares backprop with a finite-difference estimate.' }));

  function calc() {
    var x = X.get(), w = W.get(), b = B.get(), t = T.get(), v = {};
    v.p = x * w; v.z = v.p + b; v.a = DL.sigmoid(v.z); v.L = 0.5 * (v.a - t) * (v.a - t);
    v.da = v.a - t; v.sp = v.a * (1 - v.a); v.dz = v.da * v.sp; v.dp = v.dz; v.db = v.dz; v.dw = v.dp * x; v.dx = v.dp * w;
    return v;
  }
  function loss(x, w, b, t) { var a = DL.sigmoid(x * w + b); return 0.5 * (a - t) * (a - t); }
  var MSG = [
    'Press Next step. Leaves x, w, b, t are set by the sliders.',
    '<b>Forward 1:</b> p = x·w. Cache x and w; the backward pass will need them.',
    '<b>Forward 2:</b> z = p + b.',
    '<b>Forward 3:</b> a = σ(z). Cache a (since σ′ = a(1 − a)).',
    '<b>Forward 4:</b> L = ½(a − t)². The forward pass is complete.',
    '<b>Backward 5:</b> seed ∂L/∂L = 1, then ∂L/∂a = a − t.',
    '<b>Backward 6:</b> ∂L/∂z = ∂L/∂a · σ′(z) = (a − t) · a(1 − a).',
    '<b>Backward 7:</b> addition copies the gradient: ∂L/∂p = ∂L/∂z and ∂L/∂b = ∂L/∂z.',
    '<b>Backward 8:</b> multiplication swaps the other operand: ∂L/∂w = x · ∂L/∂p, ∂L/∂x = w · ∂L/∂p. Done: one sweep gave every gradient.'
  ];
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), ctx = c.ctx, v = calc(), w = c.w, h = c.h, s = step;
    ctx.clearRect(0, 0, w, h);
    var fs = Math.max(11, Math.min(13, w / 55));
    var pos = {
      x: [0.08, 0.14], w: [0.08, 0.36], b: [0.08, 0.60], t: [0.08, 0.84],
      p: [0.28, 0.25], z: [0.46, 0.48], a: [0.64, 0.48], L: [0.84, 0.66]
    };
    function PX(k) { return pos[k][0] * w; } function PY(k) { return pos[k][1] * h; }
    var R = Math.min(24, w / 22);
    function arrow(a, b, color, wd, off) {
      var x1 = PX(a), y1 = PY(a), x2 = PX(b), y2 = PY(b), ang = Math.atan2(y2 - y1, x2 - x1), r1 = ('xwbt'.indexOf(a) >= 0) ? 30 : R + 3, r2 = ('xwbt'.indexOf(b) >= 0) ? 30 : R + 3;
      var o = off || 0, ox = -Math.sin(ang) * o, oy = Math.cos(ang) * o;
      x1 += Math.cos(ang) * r1 + ox; y1 += Math.sin(ang) * r1 + oy; x2 -= Math.cos(ang) * r2 - ox; y2 -= Math.sin(ang) * r2 - oy;
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = wd; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - 8 * Math.cos(ang - 0.4), y2 - 8 * Math.sin(ang - 0.4)); ctx.lineTo(x2 - 8 * Math.cos(ang + 0.4), y2 - 8 * Math.sin(ang + 0.4)); ctx.fill();
    }
    var edges = [['x', 'p'], ['w', 'p'], ['p', 'z'], ['b', 'z'], ['z', 'a'], ['a', 'L'], ['t', 'L']];
    edges.forEach(function (e) { arrow(e[0], e[1], col.line, 1.5, 0); });
    var back = { 'a-L': 5, 'z-a': 6, 'p-z': 7, 'b-z': 7, 'x-p': 8, 'w-p': 8 };
    edges.forEach(function (e) { var k = e[0] + '-' + e[1]; if (back[k] && s >= back[k]) arrow(e[1], e[0], col.accent, 2.2, 5); });
    function node(k, label, shape, fv, gv, fshow, gshow) {
      var x = PX(k), y = PY(k);
      ctx.fillStyle = col.soft; ctx.strokeStyle = col.muted; ctx.lineWidth = 1.5; ctx.beginPath();
      if (shape === 'box') { ctx.rect(x - 28, y - 14, 56, 28); } else ctx.arc(x, y, R, 0, 7);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = col.ink; ctx.font = fs + 'px monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(label, x, y);
      if (fshow) { ctx.fillStyle = col.ink; ctx.font = 'bold ' + fs + 'px monospace'; ctx.fillText(fv, x, y - (shape === 'box' ? 25 : R + 12)); }
      if (gshow) { ctx.fillStyle = col.accent; ctx.font = 'bold ' + fs + 'px monospace'; ctx.fillText(gv, x, y + (shape === 'box' ? 27 : R + 13)); }
    }
    var f4 = function (q) { return (+q).toFixed(4); };
    node('x', 'x', 'box', f4(X.get()), 'dx=' + f4(v.dx), true, s >= 8);
    node('w', 'w', 'box', f4(W.get()), 'dw=' + f4(v.dw), true, s >= 8);
    node('b', 'b', 'box', f4(B.get()), 'db=' + f4(v.db), true, s >= 7);
    node('t', 't', 'box', f4(T.get()), '', true, false);
    node('p', '×', 'c', 'p=' + f4(v.p), 'dp=' + f4(v.dp), s >= 1, s >= 7);
    node('z', '+', 'c', 'z=' + f4(v.z), 'dz=' + f4(v.dz), s >= 2, s >= 6);
    node('a', 'σ', 'c', 'a=' + f4(v.a), 'da=' + f4(v.da), s >= 3, s >= 5);
    node('L', 'L', 'c', 'L=' + f4(v.L), 'dL=1', s >= 4, s >= 5);
    ctx.fillStyle = col.muted; ctx.font = '12px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('step ' + s + ' / ' + MAXS + (s === 0 ? '' : s <= 4 ? '  (forward)' : '  (backward)') + '    d* means ∂L/∂*', 10, h - 8);
    var e = 1e-5, nw = (loss(X.get(), W.get() + e, B.get(), T.get()) - loss(X.get(), W.get() - e, B.get(), T.get())) / (2 * e);
    var nb = (loss(X.get(), W.get(), B.get() + e, T.get()) - loss(X.get(), W.get(), B.get() - e, T.get())) / (2 * e);
    say(MSG[s] + (s >= 8 ? '<br>Check: backprop ∂L/∂w = <b>' + f4(v.dw) + '</b>, finite difference = <b>' + f4(nw) + '</b>; ∂L/∂b = <b>' + f4(v.db) + '</b> vs <b>' + f4(nb) + '</b>.' : ''));
  }
  c.draw = draw;
});
