DL.register('week05-conv-explorer', function (root) {
  var N = 10;
  var presets = {
    identity: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    vedge: [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
    hedge: [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
    blur: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
    sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    laplace: [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    shift: [[0, 0, 0], [0, 0, 1], [0, 0, 0]]
  };
  var images = {
    box: function (i, j) { return i >= 2 && i <= 7 && j >= 2 && j <= 7 ? 1 : 0; },
    diag: function (i, j) { return j >= i - 1 && j <= i + 1 ? 1 : 0; },
    letter: function (i, j) { return (j === 3 && i >= 1 && i <= 8) || (i === 1 && j >= 3 && j <= 6) || (i === 4 && j >= 3 && j <= 6) || (j === 6 && (i === 2 || i === 3)) ? 1 : 0; },
    ramp: function (i, j) { return j / (N - 1); },
    blank: function () { return 0; }
  };
  var X = [], K = [[0, 0, 0], [0, 1, 0], [0, 0, 0]];
  function loadImage(name) { for (var i = 0; i < N; i++) { X[i] = []; for (var j = 0; j < N; j++) X[i][j] = images[name](i, j); } }
  loadImage('letter');

  var ctrl = DL.controls(root);
  DL.select(ctrl, { label: 'Image', value: 'letter', options: [{ value: 'letter', label: 'letter P' }, { value: 'box', label: 'square' }, { value: 'diag', label: 'diagonal' }, { value: 'ramp', label: 'ramp' }, { value: 'blank', label: 'blank (paint it)' }] }, function (v) { loadImage(v); draw(); });
  var kernSel = DL.select(ctrl, { label: 'Kernel preset', value: 'vedge', options: [{ value: 'identity', label: 'identity' }, { value: 'vedge', label: 'Sobel vertical edges' }, { value: 'hedge', label: 'Sobel horizontal edges' }, { value: 'blur', label: 'box blur' }, { value: 'sharpen', label: 'sharpen' }, { value: 'laplace', label: 'Laplacian' }, { value: 'shift', label: 'shift (reads right neighbor)' }] }, function (v) { setK(presets[v]); });
  var mode = DL.select(ctrl, { label: 'Operation', value: 'xcorr', options: [{ value: 'xcorr', label: 'cross-correlation (deep learning)' }, { value: 'conv', label: 'true convolution (flipped kernel)' }] }, draw);
  var stride = DL.slider(ctrl, { label: 'Stride', min: 1, max: 3, step: 1, value: 1 }, draw);
  var pad = DL.slider(ctrl, { label: 'Padding', min: 0, max: 2, step: 1, value: 0 }, draw);
  var dil = DL.slider(ctrl, { label: 'Dilation', min: 1, max: 3, step: 1, value: 1 }, draw);
  var pos = DL.slider(ctrl, { label: 'Output position', min: 0, max: 63, step: 1, value: 0 }, draw);

  var kRow = DL.row(root);
  var kin = [];
  kRow.appendChild(DL.el('span', { class: 'dl-slider-label', text: 'Kernel (editable): ' }));
  for (var a = 0; a < 3; a++) for (var b = 0; b < 3; b++) (function (a, b) {
    var inp = DL.el('input', { type: 'number', step: 'any', value: 0, style: { width: '58px' } });
    inp.addEventListener('input', function () { var v = parseFloat(inp.value); K[a][b] = isFinite(v) ? v : 0; draw(); });
    kin.push(inp); kRow.appendChild(inp);
  })(a, b);
  function setK(M) { K = M.map(function (r) { return r.slice(); }); for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++) kin[i * 3 + j].value = +K[i][j].toFixed(4); draw(); }
  var c = DL.canvas(root, { aspect: 0.42 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', html: 'Try: click input pixels to paint them; pick Sobel and watch vertical strokes light up; switch to true convolution and see the output change for asymmetric kernels (shift); raise dilation to widen the footprint.' }));

  var layout = {};
  function compute() {
    var s = Math.round(stride.get()), p = Math.round(pad.get()), d = Math.round(dil.get());
    var keff = d * 2 + 1, no = Math.floor((N + 2 * p - keff) / s) + 1;
    var flip = mode.get() === 'conv';
    var Y = [];
    for (var i = 0; i < Math.max(no, 0); i++) { Y[i] = []; for (var j = 0; j < no; j++) {
      var sum = 0;
      for (var u = 0; u < 3; u++) for (var v = 0; v < 3; v++) {
        var r = i * s - p + u * d, q = j * s - p + v * d;
        var w = flip ? K[2 - u][2 - v] : K[u][v];
        if (r >= 0 && r < N && q >= 0 && q < N) sum += w * X[r][q];
      }
      Y[i][j] = sum;
    } }
    return { Y: Y, no: no, s: s, p: p, d: d, keff: keff, flip: flip };
  }
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), ctx = c.ctx, R = compute();
    ctx.clearRect(0, 0, c.w, c.h);
    var pw = c.w * 0.3, top = 28, gap = (c.w - 3 * pw) / 4;
    var xs = [gap, 2 * gap + pw, 3 * gap + 2 * pw];
    layout = { x0: xs[0], y0: top, pw: pw };
    ctx.fillStyle = col.ink; ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('input (' + N + 'x' + N + ', click to paint)', xs[0] + pw / 2, 16);
    ctx.fillText('kernel', xs[1] + pw / 2, 16);
    ctx.fillText('output (' + Math.max(R.no, 0) + 'x' + Math.max(R.no, 0) + ')', xs[2] + pw / 2, 16);
    DL.heatmap(ctx, X, xs[0], top, pw, pw, { min: 0, max: 1, cmap: DL.cmap.gray, labels: false });
    var kk = R.flip ? K.map(function (r) { return r.slice().reverse(); }).reverse() : K;
    DL.heatmap(ctx, kk, xs[1] + pw * 0.15, top + pw * 0.15, pw * 0.7, pw * 0.7, { min: -2, max: 2, cmap: DL.cmap.diverge, labels: true, fmt: function (v) { return (+v.toFixed(2)) + ''; } });
    if (R.no < 1) { say('Kernel footprint ' + R.keff + 'x' + R.keff + ' exceeds the padded input: no valid output positions.'); return; }
    var mx = 0; R.Y.forEach(function (r) { r.forEach(function (v) { mx = Math.max(mx, Math.abs(v)); }); }); mx = mx || 1;
    var cw = pw / R.no;
    DL.heatmap(ctx, R.Y, xs[2], top, pw, pw, { min: -mx, max: mx, cmap: DL.cmap.diverge, labels: R.no <= 8, fmt: function (v) { return (+v.toFixed(1)) + ''; } });
    var tot = R.no * R.no, idx = Math.min(Math.round(pos.get()), tot - 1);
    var oi = Math.floor(idx / R.no), oj = idx % R.no;
    var cell = pw / N;
    ctx.strokeStyle = col.accent; ctx.lineWidth = 2;
    for (var u = 0; u < 3; u++) for (var v = 0; v < 3; v++) {
      var r = oi * R.s - R.p + u * R.d, q = oj * R.s - R.p + v * R.d;
      if (r >= 0 && r < N && q >= 0 && q < N) ctx.strokeRect(xs[0] + q * cell + 1, top + r * cell + 1, cell - 2, cell - 2);
    }
    ctx.strokeRect(xs[2] + oj * cw + 1, top + oi * cw + 1, cw - 2, cw - 2);
    say('Output size = floor((' + N + ' + 2*' + R.p + ' - ' + R.keff + ')/' + R.s + ') + 1 = <b>' + R.no + '</b> (effective kernel ' + R.keff + 'x' + R.keff + '). Selected y[' + oi + ',' + oj + '] = <b>' + DL.fmt(R.Y[oi][oj], 2) + '</b>. Parameters: 9, independent of image size.');
    layout.cell = cell;
  }
  DL.drag(c.canvas, {
    down: function (x, y) { paint(x, y); }, move: function (x, y) { paint(x, y); }
  });
  function paint(x, y) {
    if (!layout.pw) return;
    var j = Math.floor((x - layout.x0) / (layout.pw / N)), i = Math.floor((y - layout.y0) / (layout.pw / N));
    if (i >= 0 && i < N && j >= 0 && j < N && X[i][j] !== 1) { X[i][j] = 1; draw(); }
  }
  c.canvas.addEventListener('dblclick', function (e) {
    var r = c.canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    var j = Math.floor((x - layout.x0) / (layout.pw / N)), i = Math.floor((y - layout.y0) / (layout.pw / N));
    if (i >= 0 && i < N && j >= 0 && j < N) { X[i][j] = 0; draw(); }
  });
  c.draw = draw;
  setK(presets.vedge);
  var hint = root.querySelector('.dl-hint'); if (hint) hint.innerHTML += ' Drag paints white; double-click erases a pixel.';
});
