DL.register('week05-equivariance', function (root) {
  var N = 48, rand = DL.rng(7), base = [], i;
  for (i = 0; i < N; i++) base.push(0);
  // a smooth bump plus a small distractor bump
  function bump(c, w, a) { for (var t = 0; t < N; t++) base[t] += a * Math.exp(-Math.pow((t - c) / w, 2)); }
  bump(14, 2.2, 1); bump(24, 1.2, 0.5);
  var ctrl = DL.controls(root);
  var shift = DL.slider(ctrl, { label: 'Shift input by', min: 0, max: 12, step: 1, value: 0 }, draw);
  var pool = DL.slider(ctrl, { label: 'Pool window (stride = window)', min: 1, max: 8, step: 1, value: 4 }, draw);
  var kind = DL.select(ctrl, { label: 'Kernel', value: 'bump', options: [{ value: 'bump', label: 'bump detector' }, { value: 'edge', label: 'edge [-1, 0, 1]' }, { value: 'avg', label: 'smoother' }] }, draw);
  var c = DL.canvas(root, { aspect: 0.62 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Top: input (dashed = unshifted). Middle: conv response, which moves with the input (equivariance). Bottom: strided max-pool output upsampled for display; it only moves in whole windows, so small shifts change it (approximate invariance at best). Global max pooling would be exactly invariant to shifts that keep the bump in view.' }));
  function kernel() {
    var t = kind.get();
    if (t === 'edge') return [-1, 0, 1];
    if (t === 'avg') return [0.25, 0.5, 0.25];
    return [0.1, 0.4, 0.8, 0.4, 0.1];
  }
  function shifted(x, s) { var y = []; for (var t = 0; t < N; t++) y.push(t - s >= 0 ? x[t - s] : 0); return y; }
  function conv(x, w) { var y = [], h = Math.floor(w.length / 2); for (var t = 0; t < N; t++) { var s = 0; for (var j = 0; j < w.length; j++) { var q = t + j - h; if (q >= 0 && q < N) s += w[j] * x[q]; } y.push(s); } return y; }
  function mpool(x, k) { var out = []; for (var t = 0; t + k <= N; t += k) out.push(Math.max.apply(null, x.slice(t, t + k))); return out; }
  function draw() {
    if (!c.w) return;
    var s = Math.round(shift.get()), k = Math.round(pool.get()), w = kernel(), col = DL.colors(), ctx = c.ctx;
    var xs = shifted(base, s), y0 = conv(base, w), y1 = conv(xs, w), y0s = shifted(y0, s);
    var p0 = mpool(y0, k), p1 = mpool(y1, k);
    var mx = 0, mn = 0; y0.concat(y1).forEach(function (v) { mx = Math.max(mx, v); mn = Math.min(mn, v); });
    ctx.clearRect(0, 0, c.w, c.h);
    var panels = [
      { title: 'input x', a: base, b: xs, lo: -0.1, hi: 1.2 },
      { title: 'conv response y = w * x', a: y0s, b: y1, lo: Math.min(mn, -0.1) * 1.1, hi: mx * 1.1 + 0.01 },
      { title: 'max-pool, window ' + k + ' (shifted: orange, original: gray)', a: null, b: null, lo: Math.min(mn, -0.1) * 1.1, hi: mx * 1.1 + 0.01 }
    ];
    var ph = c.h / 3, L = 12, W = c.w - 24;
    panels.forEach(function (P, idx) {
      var y = idx * ph + 20, h = ph - 30;
      function X(t) { return L + (t + 0.5) / N * W; }
      function Y(v) { return y + h - (v - P.lo) / (P.hi - P.lo) * h; }
      ctx.fillStyle = col.muted; ctx.font = '12px sans-serif'; ctx.textAlign = 'left'; ctx.fillText(P.title, L, y - 6);
      ctx.strokeStyle = col.line; ctx.strokeRect(L, y, W, h);
      function line(arr, color, wd, dash) { ctx.strokeStyle = color; ctx.lineWidth = wd; ctx.setLineDash(dash || []); ctx.beginPath(); arr.forEach(function (v, t) { t ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v)); }); ctx.stroke(); ctx.setLineDash([]); }
      function steps(arr, color, wd) { ctx.strokeStyle = color; ctx.lineWidth = wd; ctx.beginPath(); arr.forEach(function (v, q) { var x0 = L + q * k / N * W, x1 = L + (q + 1) * k / N * W; q ? ctx.lineTo(x0, Y(v)) : ctx.moveTo(x0, Y(v)); ctx.lineTo(x1, Y(v)); }); ctx.stroke(); }
      if (P.a) { line(P.a, col.muted, 1.5, [5, 4]); line(P.b, col.accent, 2.5); }
      else { steps(p0, col.muted, 1.5); steps(p1, col.accent, 2.5); }
    });
    var errConv = 0, ref = 0; for (var t = 6; t < N - 6; t++) { errConv += Math.abs(y1[t] - y0s[t]); ref += Math.abs(y0s[t]); }
    var d = 0, m = Math.min(p0.length, p1.length); for (t = 0; t < m; t++) d += Math.abs(p1[t] - p0[t]);
    var g0 = Math.max.apply(null, y0), g1 = Math.max.apply(null, y1);
    say('Equivariance error (interior) |conv(shift x) - shift(conv x)| = <b>' + DL.fmt(errConv, 6) + '</b>. Pooled outputs differ by <b>' + DL.fmt(d, 3) + '</b> (L1, un-shifted comparison). Global max: <b>' + DL.fmt(g0, 3) + '</b> vs <b>' + DL.fmt(g1, 3) + '</b>' + (Math.abs(g0 - g1) < 1e-9 ? ' (identical: invariant)' : '') + '.');
  }
  c.draw = draw;
});
