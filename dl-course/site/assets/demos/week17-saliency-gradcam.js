DL.register('week17-saliency-gradcam', function (root) {
  // A tiny fixed "CNN" on a 32x32 synthetic image:
  //   conv 5x5 (3 hand-made channels: blob, vertical edge, horizontal edge) -> ReLU -> 4x4 average pool (8x8 map)
  //   -> global average pool -> weights w_k -> tanh(gain * s) = class score.
  // Gradients are computed analytically (manual backprop), so saliency, SmoothGrad and Grad-CAM are exact.
  var S = 32, KS = 5, H = 8;
  var ctrl = DL.controls(root);
  var method = DL.select(ctrl, { label: 'Method', value: 'gradcam', options: [
    { value: 'saliency', label: 'Gradient saliency |dy/dx|' }, { value: 'smooth', label: 'SmoothGrad' }, { value: 'gradcam', label: 'Grad-CAM (8x8 map)' }] }, draw);
  var wb = DL.slider(ctrl, { label: 'Class weight: blob', min: -2, max: 2, step: 0.1, value: 1.5, format: function (v) { return DL.fmt(v, 1); } }, draw);
  var wv = DL.slider(ctrl, { label: 'Class weight: vertical stripes', min: -2, max: 2, step: 0.1, value: 0, format: function (v) { return DL.fmt(v, 1); } }, draw);
  var gain = DL.slider(ctrl, { label: 'Output gain (saturation)', min: 0.1, max: 60, step: 0.1, value: 1, format: function (v) { return DL.fmt(v, 1); } }, draw);
  var sg = DL.slider(ctrl, { label: 'SmoothGrad σ', min: 0, max: 0.6, step: 0.02, value: 0.2, format: function (v) { return DL.fmt(v, 2); } }, draw);
  var c = DL.canvas(root, { aspect: 0.36 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Left: input with a bright blob (top-left) and vertical stripes (bottom-right). Middle: attribution map for the "class" whose score uses the weights above. Right: map over the image. With weight on "blob" only, Grad-CAM lights the blob. Give "stripes" weight and both appear. Raise the output gain: the score saturates, the raw gradient shrinks toward zero (see max |grad| in the readout) and saliency maps degrade, whereas Grad-CAM after normalization is unaffected. Increase SmoothGrad σ to denoise the pixel map.' }));

  function makeImage() {
    var im = [], r = DL.rng(7), i, j;
    for (i = 0; i < S; i++) { im.push([]); for (j = 0; j < S; j++) {
      var v = 0.08 * (r() - 0.5);
      v += Math.exp(-(Math.pow(i - 9, 2) + Math.pow(j - 9, 2)) / 14);
      if (i >= 17 && i < 29 && j >= 17 && j < 29 && (j % 4 < 2)) v += 0.8;
      im[i].push(v);
    } }
    return im;
  }
  var IMG = makeImage();
  var filt = [];  // 3 filters 5x5
  (function () {
    var b = [], v = [], h = [], i, j;
    for (i = 0; i < KS; i++) { b.push([]); v.push([]); h.push([]); for (j = 0; j < KS; j++) {
      var d2 = Math.pow(i - 2, 2) + Math.pow(j - 2, 2);
      b[i].push(Math.exp(-d2 / 2) - 0.5 * Math.exp(-d2 / 8));
      v[i].push((j - 2) * Math.exp(-d2 / 6) * 0.5);
      h[i].push((i - 2) * Math.exp(-d2 / 6) * 0.5);
    } }
    // vertical stripe detector: alternating column filter (matches period-4 stripes)
    for (i = 0; i < KS; i++) for (j = 0; j < KS; j++) v[i][j] = Math.cos(Math.PI * (j - 2) / 2) * Math.exp(-Math.pow(i - 2, 2) / 6) * 0.35;
    filt = [b, v, h];
  })();

  function forward(im, w, g) {
    var pre = [], A = [], k, i, j, a, b2;
    for (k = 0; k < 3; k++) {
      pre.push([]); A.push([]);
      for (i = 0; i < S; i++) { pre[k].push([]); A[k].push([]); for (j = 0; j < S; j++) {
        var s = 0;
        for (a = 0; a < KS; a++) for (b2 = 0; b2 < KS; b2++) { var ii = i + a - 2, jj = j + b2 - 2; if (ii >= 0 && ii < S && jj >= 0 && jj < S) s += filt[k][a][b2] * im[ii][jj]; }
        pre[k][i].push(s); A[k][i].push(Math.max(0, s));
      } }
    }
    var pool = [], F = [];
    for (k = 0; k < 3; k++) {
      pool.push([]); var tot = 0;
      for (i = 0; i < H; i++) { pool[k].push([]); for (j = 0; j < H; j++) {
        var p = 0; for (a = 0; a < 4; a++) for (b2 = 0; b2 < 4; b2++) p += A[k][4 * i + a][4 * j + b2];
        p /= 16; pool[k][i].push(p); tot += p;
      } }
      F.push(tot / (H * H));
    }
    var sc = w[0] * F[0] + w[1] * F[1] + w[2] * F[2], y = Math.tanh(g * sc);
    return { pre: pre, A: A, pool: pool, F: F, s: sc, y: y, c: g * (1 - y * y) };
  }
  function inputGrad(im, w, g) {
    var f = forward(im, w, g), gx = [], i, j, k, a, b2;
    for (i = 0; i < S; i++) { gx.push([]); for (j = 0; j < S; j++) gx[i].push(0); }
    for (k = 0; k < 3; k++) {
      var gA = f.c * w[k] / (H * H) / 16;     // dy/dA_k(i,j), constant over positions
      for (i = 0; i < S; i++) for (j = 0; j < S; j++) {
        if (f.pre[k][i][j] <= 0) continue;
        for (a = 0; a < KS; a++) for (b2 = 0; b2 < KS; b2++) { var ii = i + a - 2, jj = j + b2 - 2; if (ii >= 0 && ii < S && jj >= 0 && jj < S) gx[ii][jj] += gA * filt[k][a][b2]; }
      }
    }
    return { g: gx, f: f };
  }
  var noiseSeed = 11;
  function maps() {
    var w = [wb.get(), wv.get(), 0], g = gain.get(), base = inputGrad(IMG, w, g), M = [], i, j, mx = 0;
    var raw = 0;
    if (method.get() === 'gradcam') {
      var f = base.f, al = [];
      for (var k = 0; k < 3; k++) al.push(f.c * w[k] / (H * H));      // alpha_k = mean over 8x8 of dy/dA_pooled
      for (i = 0; i < H; i++) { M.push([]); for (j = 0; j < H; j++) { var s = 0; for (k = 0; k < 3; k++) s += al[k] * f.pool[k][i][j]; M[i].push(Math.max(0, s)); if (M[i][j] > mx) mx = M[i][j]; } }
      raw = Math.abs(f.c);
      return { M: M, mx: mx, n: H, raw: raw, f: f };
    }
    var G = base.g;
    if (method.get() === 'smooth' && sg.get() > 0) {
      var r = DL.rng(noiseSeed), N = 20, acc = [];
      for (i = 0; i < S; i++) { acc.push([]); for (j = 0; j < S; j++) acc[i].push(0); }
      for (var n = 0; n < N; n++) {
        var im = IMG.map(function (row) { return row.map(function (v) { return v + sg.get() * DL.randn(r); }); });
        var gg = inputGrad(im, w, g).g;
        for (i = 0; i < S; i++) for (j = 0; j < S; j++) acc[i][j] += gg[i][j] / N;
      }
      G = acc;
    }
    for (i = 0; i < S; i++) { M.push([]); for (j = 0; j < S; j++) { var v = Math.abs(G[i][j]); M[i].push(v); if (v > mx) mx = v; raw = Math.max(raw, v); } }
    return { M: M, mx: mx, n: S, raw: raw, f: base.f };
  }
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), ctx = c.ctx, W = c.w, Hh = c.h, res = maps(), gap = 10;
    var size = Math.min((W - 4 * gap) / 3, Hh - 28), x0 = gap, y0 = 22, i, j;
    ctx.clearRect(0, 0, W, Hh);
    ctx.fillStyle = col.muted; ctx.font = '11px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('input', x0 + size / 2, y0 - 4); ctx.fillText('attribution map', x0 * 2 + size * 1.5, y0 - 4); ctx.fillText('overlay', x0 * 3 + size * 2.5, y0 - 4);
    var cs = size / S;
    function img(ox, alpha) {
      for (i = 0; i < S; i++) for (j = 0; j < S; j++) { ctx.globalAlpha = alpha; ctx.fillStyle = DL.cmap.gray(Math.min(1, Math.max(0, IMG[i][j]))); ctx.fillRect(ox + j * cs, y0 + i * cs, cs + 0.5, cs + 0.5); }
      ctx.globalAlpha = 1;
    }
    function map(ox, alpha) {
      var cell = size / res.n, den = res.mx > 1e-15 ? res.mx : 1;
      for (i = 0; i < res.n; i++) for (j = 0; j < res.n; j++) { ctx.globalAlpha = alpha(res.M[i][j] / den); ctx.fillStyle = DL.cmap.magma(res.M[i][j] / den); ctx.fillRect(ox + j * cell, y0 + i * cell, cell + 0.5, cell + 0.5); }
      ctx.globalAlpha = 1;
    }
    img(x0, 1);
    ctx.fillStyle = '#000'; ctx.fillRect(x0 * 2 + size, y0, size, size); map(x0 * 2 + size, function () { return 1; });
    img(x0 * 3 + size * 2, 1); map(x0 * 3 + size * 2, function (t) { return 0.7 * t; });
    ctx.strokeStyle = col.line; for (var q = 0; q < 3; q++) ctx.strokeRect(x0 * (q + 1) + size * q, y0, size, size);
    say('score y = tanh(gain · s) = <b>' + DL.fmt(res.f.y, 3) + '</b> &nbsp; s = ' + DL.fmt(res.f.s, 3) + ' &nbsp; saturation factor 1 − y² = ' + DL.fmt(1 - res.f.y * res.f.y, 4) + ' &nbsp; ' + (method.get() === 'gradcam' ? 'Grad-CAM weights α<sub>k</sub> ∝ w<sub>k</sub>·(1−y²)·gain (the map is normalized, so the scale cancels)' : 'max |grad| = <b>' + res.raw.toExponential(2) + '</b> (before normalization)'));
  }
  c.draw = draw;
  draw();
});
