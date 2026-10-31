DL.register('week13-cfg-explorer', function (root) {
  var ctrl = DL.controls(root);
  var w = DL.slider(ctrl, { label: 'guidance w', min: 0, max: 8, step: 0.1, value: 1 }, draw);
  var sep = DL.slider(ctrl, { label: 'class separation', min: 0.5, max: 4, step: 0.05, value: 2 }, draw);
  var sg = DL.slider(ctrl, { label: 'class std σ', min: 0.3, max: 1.5, step: 0.05, value: 0.8 }, draw);
  var cls = DL.select(ctrl, { label: 'Condition on', value: '1', options: [{ value: '1', label: 'class B (right)' }, { value: '0', label: 'class A (left)' }] }, draw);
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Two classes A and B with equal priors, Gaussian class-conditionals p(x|A) = N(−s/2, σ²), p(x|B) = N(+s/2, σ²). Classifier-free guidance samples (in the idealized exact-score limit) from the density ∝ p(x) · [p(c|x)]^w, equivalently p(x|c)^w p(x)^(1−w). w = 0 is the unconditional mixture, w = 1 the true conditional, w > 1 sharpens toward the class and shrinks diversity. Dashed gray: the unconditional density p(x). Overlapping classes (low separation, high σ) show the effect most clearly.' }));
  var XMIN = -7, XMAX = 7, NG = 701, xs = DL.linspace(XMIN, XMAX, NG), dx = (XMAX - XMIN) / (NG - 1);
  function g(x, m, s) { return Math.exp(-0.5 * (x - m) * (x - m) / (s * s)) / (s * 2.5066282746); }
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), s = sep.get(), sd = sg.get(), k = parseInt(cls.get(), 10), ww = w.get();
    var mu = [-s / 2, s / 2], i;
    var pu = [], pc = [], pg = [], Z = 0;
    for (i = 0; i < NG; i++) {
      var x = xs[i], a = g(x, mu[0], sd), b = g(x, mu[1], sd), u = 0.5 * (a + b), cc = k === 0 ? a : b;
      pu.push(u); pc.push(cc);
      // p(x) * p(c|x)^w, with p(c|x) = 0.5*cc/u
      var post = u > 1e-300 ? 0.5 * cc / u : 0;
      var val = u * Math.pow(Math.max(post, 0), ww); pg.push(val); Z += val * dx;
    }
    pg = pg.map(function (v) { return v / Z; });
    var m = 0, v2 = 0, pm = 0, tot = 0;
    for (i = 0; i < NG; i++) { m += xs[i] * pg[i] * dx; }
    for (i = 0; i < NG; i++) { v2 += (xs[i] - m) * (xs[i] - m) * pg[i] * dx; }
    // posterior class probability mass: fraction of guided samples where p(c|x) > 0.5 -> correct side
    for (i = 0; i < NG; i++) { var uu = pu[i]; var po = uu > 1e-300 ? 0.5 * pc[i] / uu : 0; if (po > 0.5) pm += pg[i] * dx; tot += pg[i] * dx; }
    var ymax = Math.max(0.6, Math.max.apply(null, pg) * 1.15, Math.max.apply(null, pu) * 1.15);
    var P = DL.plot(c, { xmin: XMIN, xmax: XMAX, ymin: 0, ymax: ymax, xlabel: 'x', ylabel: 'density' });
    P.clear(); P.grid(); P.axes();
    P.line(xs, xs.map(function (x) { return 0.5 * g(x, mu[0], sd); }), col.blue, 1.2, [2, 3]);
    P.line(xs, xs.map(function (x) { return 0.5 * g(x, mu[1], sd); }), col.green, 1.2, [2, 3]);
    P.line(xs, pu, col.muted, 1.8, [6, 4]);
    P.line(xs, pc, col.purple, 2, [1, 0]);
    P.line(xs, pg, col.accent, 3);
    P.text('p(x) unconditional', XMIN + 0.3, ymax * 0.95, col.muted, 'left');
    P.text('p(x | c)', XMIN + 0.3, ymax * 0.87, col.purple, 'left');
    P.text('guided (w = ' + DL.fmt(ww, 1) + ')', XMIN + 0.3, ymax * 0.79, col.accent, 'left');
    var truth = k === 0 ? mu[0] : mu[1];
    say('guided mean = <b>' + DL.fmt(m, 3) + '</b> (true class mean ' + DL.fmt(truth, 3) + ') &nbsp; guided std = <b>' + DL.fmt(Math.sqrt(v2), 3) + '</b> (class std ' + DL.fmt(sd, 2) + ') &nbsp; mass on the correct side of the decision boundary: <b>' + DL.fmt(100 * pm / Math.max(tot, 1e-12), 1) + '%</b>');
  }
  c.draw = draw;
});
