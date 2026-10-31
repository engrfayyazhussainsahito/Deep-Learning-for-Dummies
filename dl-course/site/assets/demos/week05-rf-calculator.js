DL.register('week05-rf-calculator', function (root) {
  var presets = {
    vgg: { name: 'VGG-16 conv stack', size: 224, layers: [[3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [2, 2, 1, 'pool'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [2, 2, 1, 'pool'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [2, 2, 1, 'pool'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [2, 2, 1, 'pool'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [2, 2, 1, 'pool']] },
    plain: { name: 'Ten 3x3 convs, no downsampling', size: 64, layers: [[3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv']] },
    dilated: { name: 'Dilated stack (d = 1,2,4,8,16)', size: 128, layers: [[3, 1, 1, 'conv'], [3, 1, 2, 'conv'], [3, 1, 4, 'conv'], [3, 1, 8, 'conv'], [3, 1, 16, 'conv']] },
    alex: { name: 'AlexNet conv stack', size: 227, layers: [[11, 4, 1, 'conv'], [3, 2, 1, 'pool'], [5, 1, 1, 'conv'], [3, 2, 1, 'pool'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [3, 1, 1, 'conv'], [3, 2, 1, 'pool']] },
    empty: { name: 'Empty (build your own)', size: 64, layers: [] }
  };
  var L = presets.vgg.layers.map(function (l) { return l.slice(); });
  var ctrl = DL.controls(root);
  var sel = DL.select(ctrl, { label: 'Preset', value: 'vgg', options: Object.keys(presets).map(function (k) { return { value: k, label: presets[k].name }; }) }, function (v) {
    L = presets[v].layers.map(function (l) { return l.slice(); }); size.set(presets[v].size); draw();
  });
  var size = DL.slider(ctrl, { label: 'Input size', min: 16, max: 512, step: 1, value: 224 }, draw);
  var row = DL.row(root);
  var kS = DL.select(row, { label: 'k', value: '3', options: ['1', '3', '5', '7', '11'].map(function (v) { return { value: v, label: v }; }) });
  var sS = DL.select(row, { label: 'stride', value: '1', options: ['1', '2', '4'].map(function (v) { return { value: v, label: v }; }) });
  var dS = DL.select(row, { label: 'dilation', value: '1', options: ['1', '2', '4', '8'].map(function (v) { return { value: v, label: v }; }) });
  DL.button(row, 'Add conv', function () { L.push([+kS.get(), +sS.get(), +dS.get(), 'conv']); draw(); });
  DL.button(row, 'Add max-pool', function () { L.push([+kS.get(), +sS.get(), 1, 'pool']); draw(); });
  DL.button(row, 'Remove last', function () { L.pop(); draw(); });
  var c = DL.canvas(root, { aspect: 0.4 });
  var tbl = DL.el('div', { class: 'dl-readout', style: { overflowX: 'auto', maxHeight: '260px', overflowY: 'auto' } });
  root.appendChild(tbl);
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Rule: r_l = r_(l-1) + (k_eff - 1) * j_(l-1), j_l = j_(l-1) * s, k_eff = d(k-1)+1. Try: VGG vs the plain stack, then add stride-2 pools to a plain stack; compare with the dilated stack, whose r grows exponentially with the same 3x3 kernels.' }));

  function compute() {
    var n = Math.round(size.get()), r = 1, j = 1, start = 0.5, rows = [{ i: 0, name: 'input', k: '-', s: '-', d: '-', n: n, r: 1, j: 1 }];
    for (var i = 0; i < L.length; i++) {
      var k = L[i][0], s = L[i][1], d = L[i][2], t = L[i][3];
      var ke = d * (k - 1) + 1, p = t === 'conv' ? Math.floor((ke - 1) / 2) : 0;
      var no = Math.max(0, Math.floor((n + 2 * p - ke) / s) + 1);
      r = r + (ke - 1) * j; j = j * s; n = no;
      rows.push({ i: i + 1, name: t + (t === 'conv' ? ' ' : ' ') + k + 'x' + k, k: k, s: s, d: d, n: n, r: r, j: j });
    }
    return rows;
  }
  function draw() {
    if (!c.w) return;
    var rows = compute(), col = DL.colors(), ctx = c.ctx;
    var N0 = rows[0].n, mxr = Math.max.apply(null, rows.map(function (r) { return r.r; }));
    var P = DL.plot(c, { xmin: 0, xmax: Math.max(1, rows.length - 1), ymin: 0, ymax: Math.max(mxr, N0) * 1.08, xlabel: 'layer index', ylabel: 'receptive field (pixels)' });
    P.clear(); P.grid(); P.axes();
    P.hline(N0, col.muted);
    P.text('input size ' + N0, 0.1, N0 * 1.04, col.muted, 'left');
    P.line(rows.map(function (r) { return r.i; }), rows.map(function (r) { return r.r; }), col.accent, 2.5);
    P.points(rows.map(function (r) { return r.i; }), rows.map(function (r) { return r.r; }), col.accent, 3.5);
    var h = '<table style="width:100%;border-collapse:collapse;font-size:12px"><tr><th align="left">#</th><th align="left">layer</th><th>k</th><th>s</th><th>d</th><th>output size</th><th>RF r</th><th>jump j</th></tr>';
    rows.forEach(function (r) {
      var warn = r.n < 1 ? ' style="color:' + col.red + '"' : '';
      h += '<tr' + warn + '><td>' + r.i + '</td><td>' + r.name + '</td><td align="center">' + r.k + '</td><td align="center">' + r.s + '</td><td align="center">' + r.d + '</td><td align="center">' + r.n + '</td><td align="center"><b>' + r.r + '</b></td><td align="center">' + r.j + '</td></tr>';
    });
    tbl.innerHTML = h + '</table>';
    var last = rows[rows.length - 1];
    say('Final unit sees <b>' + last.r + '</b> x <b>' + last.r + '</b> input pixels (' + (last.r >= N0 ? 'covers the whole ' + N0 + 'px input' : DL.fmt(100 * last.r * last.r / (N0 * N0), 1) + '% of the input area') + '); output map is ' + last.n + 'x' + last.n + ', adjacent outputs are ' + last.j + ' px apart.');
  }
  c.draw = draw;
  setTimeout(draw, 0);
});
