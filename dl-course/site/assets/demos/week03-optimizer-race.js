DL.register('week03-optimizer-race', function (root) {
  var SURF = {
    bowl: { name: 'Ill-conditioned bowl (curvatures 30 and 1)', xmin: -2, xmax: 2, ymin: -2, ymax: 2, start: [-1.6, 1.5],
      f: function (x, y) { return 0.5 * (30 * x * x + y * y); }, g: function (x, y) { return [30 * x, y]; }, lr: 0.03, alr: 0.08, min: [[0, 0]], levels: [0.3, 1, 2, 4, 8, 16, 30, 50] },
    valley: { name: 'Curved valley (Rosenbrock, b = 10)', xmin: -2, xmax: 2, ymin: -1, ymax: 3, start: [-1.5, 2.2],
      f: function (x, y) { var a = 1 - x, b = y - x * x; return a * a + 10 * b * b; }, g: function (x, y) { var b = y - x * x; return [-2 * (1 - x) - 40 * x * b, 20 * b]; }, lr: 0.002, alr: 0.02, min: [[1, 1]], levels: [0.5, 1.5, 3, 6, 12, 25, 50, 100, 200] },
    saddle: { name: 'Saddle between two minima', xmin: -2, xmax: 2, ymin: -2.2, ymax: 2.2, start: [1.4, 0.02],
      f: function (x, y) { return x * x - y * y + 0.25 * y * y * y * y; }, g: function (x, y) { return [2 * x, -2 * y + y * y * y]; }, lr: 0.05, alr: 0.05, min: [[0, 1.414], [0, -1.414]], levels: [-1, -0.7, -0.3, 0, 0.5, 1, 2, 3, 4] },
    himmelblau: { name: 'Himmelblau (four minima)', xmin: -5, xmax: 5, ymin: -5, ymax: 5, start: [-0.5, 4],
      f: function (x, y) { var a = x * x + y - 11, b = x + y * y - 7; return a * a + b * b; }, g: function (x, y) { var a = x * x + y - 11, b = x + y * y - 7; return [4 * x * a + 2 * b, 2 * a + 4 * y * b]; }, lr: 0.006, alr: 0.15, min: [[3, 2], [-2.805, 3.131], [-3.779, -3.283], [3.584, -1.848]], levels: [1, 5, 15, 40, 80, 150, 250, 400] }
  };
  var OPT = [
    { id: 'sgd', name: 'SGD', col: 'muted' }, { id: 'mom', name: 'Momentum', col: 'blue' }, { id: 'nag', name: 'Nesterov', col: 'teal' },
    { id: 'adagrad', name: 'AdaGrad', col: 'amber' }, { id: 'rms', name: 'RMSProp', col: 'purple' }, { id: 'adam', name: 'Adam', col: 'red' }
  ];
  var ctrl = DL.controls(root);
  var surfSel = DL.select(ctrl, { label: 'Surface', value: 'bowl', options: Object.keys(SURF).map(function (k) { return { value: k, label: SURF[k].name }; }) }, function () { reset(true); });
  var lrS = DL.slider(ctrl, { label: 'LR (SGD family)', min: -4, max: -0.5, step: 0.02, value: Math.log10(0.03), format: function (v) { return (Math.pow(10, v)).toPrecision(2); } }, function () { reset(false); });
  var alrS = DL.slider(ctrl, { label: 'LR (adaptive)', min: -4, max: 0, step: 0.02, value: Math.log10(0.08), format: function (v) { return (Math.pow(10, v)).toPrecision(2); } }, function () { reset(false); });
  var betaS = DL.slider(ctrl, { label: 'Momentum β', min: 0, max: 0.99, step: 0.01, value: 0.9 }, function () { reset(false); });
  var row = DL.row(root);
  var tog = {};
  OPT.forEach(function (o) { tog[o.id] = DL.toggle(row, o.name, true, function () { draw(); }); });
  var running = false, visible = true, raf = 0;
  var playB = DL.button(row, 'Play', function () { running = !running; playB.textContent = running ? 'Pause' : 'Play'; if (running) loop(); });
  DL.button(row, 'Reset', function () { reset(false); });
  var speedS = DL.slider(row, { label: 'Steps/frame', min: 1, max: 8, step: 1, value: 1 }, function () {});
  var c = DL.canvas(root, { aspect: 0.62 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Click the plot to move the start point. Try the bowl with a large SGD learning rate (zigzag), then raise momentum β; on the saddle surface start very close to y = 0 to see which optimizers escape the saddle first; on Himmelblau, different optimizers pick different minima. A path that leaves the plot has diverged.' }));
  var P, S, st, steps = 0, start;
  function mkState(o) { return { x: start[0], y: start[1], vx: 0, vy: 0, mx: 0, my: 0, sx: 0, sy: 0, path: [[start[0], start[1]]], dead: false, t: 0 }; }
  function reset(newSurf) {
    S = SURF[surfSel.get()];
    if (newSurf) {
      start = S.start.slice();
      lrS.set(Math.log10(S.lr)); alrS.set(Math.log10(S.alr));
    }
    if (!start) start = S.start.slice();
    P = DL.plot(c, { xmin: S.xmin, xmax: S.xmax, ymin: S.ymin, ymax: S.ymax, xlabel: 'x', ylabel: 'y' });
    st = {}; OPT.forEach(function (o) { st[o.id] = mkState(o); }); steps = 0;
    draw();
  }
  function stepAll() {
    var lr = Math.pow(10, lrS.get()), alr = Math.pow(10, alrS.get()), b = betaS.get();
    steps++;
    OPT.forEach(function (o) {
      var s = st[o.id]; if (s.dead) return;
      var g, dx = 0, dy = 0, eps = 1e-8;
      if (o.id === 'nag') g = S.g(s.x + b * s.vx, s.y + b * s.vy); else g = S.g(s.x, s.y);
      if (o.id === 'sgd') { dx = -lr * g[0]; dy = -lr * g[1]; }
      else if (o.id === 'mom' || o.id === 'nag') { s.vx = b * s.vx - lr * g[0]; s.vy = b * s.vy - lr * g[1]; dx = s.vx; dy = s.vy; }
      else if (o.id === 'adagrad') { s.sx += g[0] * g[0]; s.sy += g[1] * g[1]; dx = -alr * g[0] / (Math.sqrt(s.sx) + eps); dy = -alr * g[1] / (Math.sqrt(s.sy) + eps); }
      else if (o.id === 'rms') { s.sx = 0.9 * s.sx + 0.1 * g[0] * g[0]; s.sy = 0.9 * s.sy + 0.1 * g[1] * g[1]; dx = -alr * g[0] / (Math.sqrt(s.sx) + eps); dy = -alr * g[1] / (Math.sqrt(s.sy) + eps); }
      else { s.t++; s.mx = 0.9 * s.mx + 0.1 * g[0]; s.my = 0.9 * s.my + 0.1 * g[1]; s.sx = 0.999 * s.sx + 0.001 * g[0] * g[0]; s.sy = 0.999 * s.sy + 0.001 * g[1] * g[1];
        var c1 = 1 - Math.pow(0.9, s.t), c2 = 1 - Math.pow(0.999, s.t);
        dx = -alr * (s.mx / c1) / (Math.sqrt(s.sx / c2) + eps); dy = -alr * (s.my / c1) / (Math.sqrt(s.sy / c2) + eps); }
      s.x += dx; s.y += dy;
      if (!isFinite(s.x) || !isFinite(s.y) || Math.abs(s.x) > 50 || Math.abs(s.y) > 50) { s.dead = true; return; }
      s.path.push([s.x, s.y]);
      if (s.path.length > 4000) s.path.shift();
    });
  }
  function draw() {
    if (!P) return;
    var col = DL.colors();
    P.clear(); P.grid(); P.axes();
    var f0 = S.f, fmin = S.id === 'saddle' ? -1 : 0;
    P.contour(function (x, y) { return f0(x, y); }, S.levels, col.line);
    S.min.forEach(function (m) { P.points([m[0]], [m[1]], col.green, 5); });
    P.points([start[0]], [start[1]], col.ink, 4);
    var out = [];
    OPT.forEach(function (o) {
      if (!tog[o.id].get()) return;
      var s = st[o.id], xs = s.path.map(function (p) { return p[0]; }), ys = s.path.map(function (p) { return p[1]; });
      P.line(xs, ys, col[o.col], 2);
      P.points([s.x], [s.y], col[o.col], 4.5);
      out.push('<span style="color:' + col[o.col] + '">● ' + o.name + '</span> f=' + (s.dead ? 'diverged' : DL.fmt(S.f(s.x, s.y), 3)));
    });
    say('step <b>' + steps + '</b> &nbsp; ' + out.join(' &nbsp; '));
  }
  c.draw = draw;
  function loop() {
    cancelAnimationFrame(raf);
    if (!running || !visible) return;
    for (var i = 0; i < speedS.get(); i++) stepAll();
    draw();
    raf = requestAnimationFrame(loop);
  }
  DL.drag(c.canvas, { down: pick, move: pick });
  function pick(px, py) {
    if (!P) return;
    var x = P.dx(px), y = P.dy(py);
    if (x < S.xmin || x > S.xmax || y < S.ymin || y > S.ymax) return;
    start = [x, y]; reset(false);
  }
  if (window.IntersectionObserver) new IntersectionObserver(function (e) { visible = e[0].isIntersecting; if (visible && running) loop(); }).observe(root);
  SURF.bowl.id = 'bowl'; SURF.saddle.id = 'saddle';
  reset(true);
});
