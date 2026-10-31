DL.register('week03-lr-schedules', function (root) {
  var ctrl = DL.controls(root);
  var T = DL.slider(ctrl, { label: 'Total steps', min: 200, max: 5000, step: 100, value: 1000 }, draw);
  var warm = DL.slider(ctrl, { label: 'Warmup fraction', min: 0, max: 0.4, step: 0.01, value: 0.05 }, draw);
  var minr = DL.slider(ctrl, { label: 'Final LR / peak', min: 0, max: 0.5, step: 0.01, value: 0.0 }, draw);
  var cyc = DL.slider(ctrl, { label: 'Cycle length / step size', min: 50, max: 1000, step: 25, value: 250 }, draw);
  var drop = DL.slider(ctrl, { label: 'Step decay factor γ', min: 0.05, max: 0.9, step: 0.05, value: 0.1 }, draw);
  var row = DL.row(root);
  var SCH = [
    { id: 'const', name: 'Constant', col: 'muted' }, { id: 'step', name: 'Step decay', col: 'amber' }, { id: 'cos', name: 'Cosine', col: 'blue' },
    { id: 'wcos', name: 'Linear warmup + cosine', col: 'red' }, { id: 'tri', name: 'Triangular cyclical', col: 'green' },
    { id: 'one', name: 'One-cycle', col: 'purple' }, { id: 'wr', name: 'Cosine warm restarts', col: 'teal' }
  ];
  var tog = {};
  SCH.forEach(function (s) { tog[s.id] = DL.toggle(row, s.name, ['wcos', 'step', 'one'].indexOf(s.id) >= 0, draw); });
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Peak LR is normalised to 1. Step decay drops by γ at 30%, 60% and 90% of training; cyclical and warm-restart schedules use the cycle length slider (for the triangular policy it is the half-cycle, i.e. the step size); one-cycle ramps up for 30% of training then anneals with a cosine to a tiny value. Compare how much total learning rate each schedule hands out (the area under the curve).' }));
  function lr(id, t, N) {
    var w = Math.max(1, Math.round(warm.get() * N)), f = minr.get(), cl = cyc.get();
    var cosv = function (u) { return f + (1 - f) * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, Math.max(0, u)))); };
    switch (id) {
      case 'const': return 1;
      case 'step': return Math.pow(drop.get(), Math.floor(t / (0.3 * N)));
      case 'cos': return cosv(t / N);
      case 'wcos': return t < w ? (t + 1) / w : cosv((t - w) / (N - w));
      case 'tri': var q = t % (2 * cl); return 0.1 + 0.9 * (q < cl ? q / cl : (2 * cl - q) / cl);
      case 'one': var u = 0.3 * N; return t < u ? 0.04 + 0.96 * t / u : 0.001 + (1 - 0.001) * 0.5 * (1 + Math.cos(Math.PI * (t - u) / (N - u)));
      case 'wr': var m = t % (2 * cl); return cosv(m / (2 * cl));
    }
    return 1;
  }
  var P;
  function draw() {
    var N = T.get(), col = DL.colors();
    P = DL.plot(c, { xmin: 0, xmax: N, ymin: 0, ymax: 1.05, xlabel: 'training step', ylabel: 'LR / peak LR' });
    P.clear(); P.grid(); P.axes();
    var info = [];
    SCH.forEach(function (s) {
      if (!tog[s.id].get()) return;
      var xs = [], ys = [], area = 0, n = Math.min(N, 600);
      for (var i = 0; i < n; i++) { var t = Math.floor(i * N / n); xs.push(t); var v = lr(s.id, t, N); ys.push(v); area += v; }
      P.line(xs, ys, col[s.col], 2.2);
      info.push('<span style="color:' + col[s.col] + '">● ' + s.name + '</span> mean LR ' + DL.fmt(area / n, 2));
    });
    say(info.join(' &nbsp; ') || 'select at least one schedule');
  }
  c.draw = draw;
  draw();
});
