DL.register('week06-compound-scaling', function (root) {
  // EfficientNet-B0 baseline: depth multiplier d scales layer counts, width w scales channels, r scales resolution.
  // FLOPs of conv nets scale (approximately) as d * w^2 * r^2.
  // columns: name, width multiplier, depth multiplier, input resolution, torchvision top-1 (or null)
  var published = [
    ['B0', 1.0, 1.0, 224, 77.69], ['B1', 1.0, 1.1, 240, 78.64], ['B2', 1.1, 1.2, 260, null], ['B3', 1.2, 1.4, 300, 82.01],
    ['B4', 1.4, 1.8, 380, null], ['B5', 1.6, 2.2, 456, null], ['B6', 1.8, 2.6, 528, null], ['B7', 2.0, 3.1, 600, 84.12]
  ];
  var ctrl = DL.controls(root);
  var phi = DL.slider(ctrl, { label: 'Compound coefficient phi', min: 0, max: 7, step: 0.1, value: 3 }, draw);
  var al = DL.slider(ctrl, { label: 'alpha (depth base)', min: 1, max: 2, step: 0.01, value: 1.2, format: function (v) { return (+v).toFixed(2); } }, draw);
  var be = DL.slider(ctrl, { label: 'beta (width base)', min: 1, max: 1.6, step: 0.01, value: 1.1, format: function (v) { return (+v).toFixed(2); } }, draw);
  var ga = DL.slider(ctrl, { label: 'gamma (resolution base)', min: 1, max: 1.6, step: 0.01, value: 1.15, format: function (v) { return (+v).toFixed(2); } }, draw);
  var row = DL.row(root);
  DL.button(row, 'Reset to paper values', function () { al.set(1.2); be.set(1.1); ga.set(1.15); phi.set(3); draw(); });
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'FLOPs ~ d * w^2 * r^2 because depth multiplies layer count, width squares through (C_in x C_out), and resolution squares through H x W. The paper constrains alpha * beta^2 * gamma^2 ~ 2, so each +1 in phi doubles FLOPs. Try scaling only one dimension (set the other two bases to 1) and compare the cost of reaching a given FLOPs budget.' }));
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), ph = phi.get(), a = al.get(), b = be.get(), g = ga.get();
    var base = a * b * b * g * g, d = Math.pow(a, ph), w = Math.pow(b, ph), r = Math.pow(g, ph), fl = d * w * w * r * r;
    var N = 71, xs = DL.linspace(0, 7, N);
    var curves = [
      { n: 'depth d = alpha^phi', f: function (x) { return Math.pow(a, x); }, c: col.blue },
      { n: 'width w = beta^phi', f: function (x) { return Math.pow(b, x); }, c: col.green },
      { n: 'resolution r = gamma^phi', f: function (x) { return Math.pow(g, x); }, c: col.purple }
    ];
    var ymax = Math.max(4, Math.min(64, Math.pow(base, 7)));
    var P = DL.plot(c, { xmin: 0, xmax: 7, ymin: 0, ymax: ymax, xlabel: 'phi', ylabel: 'multiplier relative to B0', pad: { l: 50 } });
    P.clear(); P.grid(); P.axes();
    curves.forEach(function (cv, i) { P.line(xs, xs.map(cv.f), cv.c, 2); P.text(cv.n, 0.15, ymax * (0.95 - 0.07 * i), cv.c, 'left'); });
    P.line(xs, xs.map(function (x) { return Math.pow(base, x); }), col.accent, 3);
    P.text('FLOPs = (alpha beta^2 gamma^2)^phi', 0.15, ymax * (0.95 - 0.07 * 3), col.accent, 'left');
    P.vline(ph, col.muted); P.hline(1, col.muted);
    var near = published.map(function (p) { return p; }).sort(function (u, v) { return Math.abs(Math.log(u[2] * u[1] * u[1] * (u[3] / 224) * (u[3] / 224) / fl)) - Math.abs(Math.log(v[2] * v[1] * v[1] * (v[3] / 224) * (v[3] / 224) / fl)); })[0];
    say('phi = <b>' + DL.fmt(ph, 1) + '</b>: depth x<b>' + DL.fmt(d, 2) + '</b>, width x<b>' + DL.fmt(w, 2) + '</b>, resolution x<b>' + DL.fmt(r, 2) + '</b> (' + Math.round(224 * r) + ' px). FLOPs x<b>' + DL.fmt(fl, 2) + '</b> vs B0; per unit of phi x<b>' + DL.fmt(base, 3) + '</b> (' + (Math.abs(base - 2) < 0.15 ? 'close to the paper\'s budget of 2' : 'differs from 2: the "phi" steps no longer double the cost') + ').<br>' +
      'Published family for comparison (width, depth, input): B3 = (1.2, 1.4, 300), B7 = (2.0, 3.1, 600), with B0 = (1.0, 1.0, 224). Closest published model by estimated FLOPs: <b>' + near[0] + '</b> (the released B-series coefficients only roughly follow the formula).');
  }
  c.draw = draw;
});
