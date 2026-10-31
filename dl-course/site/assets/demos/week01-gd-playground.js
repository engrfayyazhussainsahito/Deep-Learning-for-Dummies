DL.register('week01-gd-playground', function (root) {
  var SURF = {
    bowl: {
      name: 'Round bowl (kappa = 1)', f: function (x, y) { return 0.5 * (x * x + y * y); },
      g: function (x, y) { return [x, y]; }, lmax: 1, fmax: 9, fmin: 0, start: [-2.4, 1.8], lrmax: 2.4, lr: 0.5,
      levels: [0.25, 0.75, 1.5, 2.5, 4, 6, 8]
    },
    valley: {
      name: 'Narrow valley (kappa = 15)', f: function (x, y) { return 0.5 * (x * x + 15 * y * y); },
      g: function (x, y) { return [x, 15 * y]; }, lmax: 15, fmax: 12, fmin: 0, start: [-2.6, 0.7], lrmax: 0.2, lr: 0.05,
      levels: [0.25, 0.75, 1.5, 2.5, 4, 6, 8, 11]
    },
    wells: {
      name: 'Tilted double well (non-convex)', f: function (x, y) { var a = x * x - 1; return 0.25 * a * a + 0.5 * y * y + 0.15 * x; },
      g: function (x, y) { return [x * (x * x - 1) + 0.15, y]; }, lmax: 8, fmax: 6, fmin: -0.2, start: [0.3, 1.5], lrmax: 0.5, lr: 0.1,
      levels: [-0.1, 0.1, 0.3, 0.6, 1, 1.6, 2.5, 4]
    }
  };
  var key = 'valley', S = SURF[key], pos = S.start.slice(), vel = [0, 0], trail = [pos.slice()], steps = 0, running = false, diverged = false, raf = 0, last = 0, visible = true;

  var ctrl = DL.controls(root);
  var sel = DL.select(ctrl, { label: 'Surface', value: key, options: Object.keys(SURF).map(function (k) { return { value: k, label: SURF[k].name }; }) }, function (v) { key = v; S = SURF[v]; lr.set(S.lr); reset(S.start.slice()); });
  var lr = DL.slider(ctrl, { label: 'Learning rate', min: 0.001, max: 2.4, step: 0.001, value: S.lr, format: function (v) { return (+v).toFixed(3); } }, function () { update(); });
  var mom = DL.slider(ctrl, { label: 'Momentum', min: 0, max: 0.98, step: 0.01, value: 0, format: function (v) { return (+v).toFixed(2); } }, function () { update(); });
  var row = DL.row(root);
  var runBtn = DL.button(row, 'Run', function () { setRunning(!running); });
  DL.button(row, 'Step', function () { setRunning(false); step(); update(); });
  DL.button(row, 'Reset', function () { reset(S.start.slice()); });
  DL.button(row, 'Stable limit', function () { lr.set(Math.min(2.4, 1.99 / S.lmax)); update(); });
  var c = DL.canvas(root, { aspect: 0.62 });
  var P = DL.plot(c, { xmin: -3, xmax: 3, ymin: -2.2, ymax: 2.2, xlabel: 'w1', ylabel: 'w2' });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Click or drag on the plot to choose a starting point. Try the valley with a learning rate near its stability limit 2/λₘₐₓ (= 0.133): the iterate zig-zags across the narrow direction while crawling along the floor. Then add momentum. On the double well, different start points reach different minima.' }));

  function setRunning(r) {
    running = r && !diverged;
    runBtn.textContent = running ? 'Pause' : 'Run';
    cancelAnimationFrame(raf);
    if (running && visible) { last = 0; raf = requestAnimationFrame(loop); }
  }
  function reset(p) {
    setRunning(false); diverged = false; pos = p; vel = [0, 0]; trail = [pos.slice()]; steps = 0; update();
  }
  function step() {
    if (diverged) return;
    var g = S.g(pos[0], pos[1]), eta = lr.get(), m = mom.get();
    vel = [m * vel[0] - eta * g[0], m * vel[1] - eta * g[1]];
    pos = [pos[0] + vel[0], pos[1] + vel[1]];
    steps++;
    if (!isFinite(pos[0]) || !isFinite(pos[1]) || Math.abs(pos[0]) > 1e4 || Math.abs(pos[1]) > 1e4) { diverged = true; setRunning(false); }
    trail.push(pos.slice());
    if (trail.length > 3000) trail.shift();
  }
  function loop(t) {
    if (!running) return;
    if (!last || t - last > 60) { last = t; for (var i = 0; i < 2; i++) step(); update(); }
    if (steps > 20000) { setRunning(false); return; }
    raf = requestAnimationFrame(loop);
  }
  // pause when scrolled out of view
  if (window.IntersectionObserver) {
    new IntersectionObserver(function (es) {
      visible = es[0].isIntersecting;
      if (!visible) { cancelAnimationFrame(raf); } else if (running) { last = 0; raf = requestAnimationFrame(loop); }
    }).observe(root);
  }
  DL.drag(c.canvas, {
    down: function (x, y) { reset([DL.clamp(P.dx(x), -3, 3), DL.clamp(P.dy(y), -2.2, 2.2)]); },
    move: function (x, y) { reset([DL.clamp(P.dx(x), -3, 3), DL.clamp(P.dy(y), -2.2, 2.2)]); }
  });

  function update() { c.redraw(); }
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), ctx = c.ctx, f = S.f;
    P.clear();
    var span = S.fmax - S.fmin;
    P.heat(function (x, y) { return Math.sqrt(Math.max(0, f(x, y) - S.fmin) / span); }, DL.cmap.warm, 6);
    P.contour(f, S.levels, col.muted);
    P.axes();
    var xs = trail.map(function (p) { return p[0]; }), ys = trail.map(function (p) { return p[1]; });
    P.line(xs, ys, col.blue, 2);
    if (trail.length < 150) P.points(xs, ys, col.blue, 2.4);
    P.points([pos[0]], [pos[1]], col.accent, 5.5);
    P.points([trail[0][0]], [trail[0][1]], col.ink, 3.5);
    var val = f(pos[0], pos[1]), g = S.g(pos[0], pos[1]), gn = Math.sqrt(g[0] * g[0] + g[1] * g[1]);
    var eta = lr.get(), lim = 2 / S.lmax, m = mom.get();
    var verdict;
    if (diverged) verdict = ' <b style="color:' + col.red + '">DIVERGED</b> (step size too large for this curvature).';
    else if (m === 0 && eta > lim) verdict = ' lr is above 2/λₘₐₓ = ' + DL.fmt(lim, 3) + ' (unstable on a quadratic).';
    else if (m === 0 && key !== 'wells') verdict = ' lr is ' + DL.fmt(eta / lim, 2) + ' of the stability limit ' + DL.fmt(lim, 3) + '.';
    else verdict = '';
    say('step <b>' + steps + '</b> &nbsp; (w1, w2) = (<b>' + DL.fmt(pos[0], 3) + '</b>, <b>' + DL.fmt(pos[1], 3) + '</b>) &nbsp; loss = <b>' + (isFinite(val) ? DL.fmt(val, 4) : 'inf') + '</b> &nbsp; |∇L| = <b>' + (isFinite(gn) ? DL.fmt(gn, 4) : 'inf') + '</b>.' + verdict);
  }
  c.draw = draw;
});
