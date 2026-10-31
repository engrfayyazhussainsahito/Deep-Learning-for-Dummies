DL.register('week02-relu-fit', function (root) {
  var TARGETS = {
    sine: { name: 'sin(3x) + 0.3x', f: function (x) { return Math.sin(3 * x) + 0.3 * x; } },
    wave: { name: 'fast wiggle: sin(7x)', f: function (x) { return 0.8 * Math.sin(7 * x); } },
    abs: { name: '|x| - 0.5 (one kink)', f: function (x) { return Math.abs(x) - 0.5; } },
    step: { name: 'smoothed step', f: function (x) { return Math.tanh(12 * x); } },
    bump: { name: 'two Gaussian bumps', f: function (x) { return Math.exp(-Math.pow((x + 0.5) / 0.15, 2)) + 0.7 * Math.exp(-Math.pow((x - 0.4) / 0.25, 2)) - 0.3; } }
  };
  var N = 80, XS = DL.linspace(-1, 1, N), H = 8, net, tgt = 'sine', steps = 0, running = false, raf = 0, visible = true, mse = 0, rand = DL.rng(11);
  var ctrl = DL.controls(root);
  var selT = DL.select(ctrl, { label: 'Target', value: tgt, options: Object.keys(TARGETS).map(function (k) { return { value: k, label: TARGETS[k].name }; }) }, function (v) { tgt = v; init(); });
  var hs = DL.slider(ctrl, { label: 'Hidden ReLU units', min: 1, max: 40, step: 1, value: H, format: function (v) { return String(Math.round(v)); } }, function (v) { H = Math.round(v); init(); });
  var lrS = DL.slider(ctrl, { label: 'Learning rate (Adam)', min: 0.002, max: 0.05, step: 0.001, value: 0.01, format: function (v) { return (+v).toFixed(3); } }, function () {});
  var tog = DL.toggle(ctrl, 'Show individual hinges', true, function () { c.redraw(); });
  var row = DL.row(root);
  var runBtn = DL.button(row, 'Train', function () { setRun(!running); });
  DL.button(row, 'Re-initialise', function () { init(); });
  var c = DL.canvas(root, { aspect: 0.55 });
  var P = DL.plot(c, { xmin: -1, xmax: 1, ymin: -2, ymax: 2, xlabel: 'x', ylabel: 'y' });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'The model is f(x) = d + Σ cᵢ ReLU(wᵢ x + bᵢ), trained by Adam on 80 samples of the target. Faint curves are the individual hinges cᵢ ReLU(wᵢ x + bᵢ); dots mark their kinks. Try 1 unit on |x| (one hinge cannot make a V), then 3 or 4 units, then raise the count on sin(7x) and watch the error fall.' }));

  function init() {
    setRun(false);
    net = { w: [], b: [], c: [], d: 0, m: null, v: null, t: 0 };
    for (var i = 0; i < H; i++) {
      var kink = -0.95 + 1.9 * (H === 1 ? 0.5 : i / (H - 1));
      var s = rand() < 0.5 ? -1 : 1;
      net.w.push(s * (0.8 + 1.2 * rand())); net.b.push(-net.w[i] * kink); net.c.push(0.3 * DL.randn(rand));
    }
    var n = 3 * H + 1; net.m = new Array(n).fill(0); net.v = new Array(n).fill(0);
    steps = 0; c.redraw();
  }
  function predict(x) { var y = net.d; for (var i = 0; i < H; i++) { var z = net.w[i] * x + net.b[i]; if (z > 0) y += net.c[i] * z; } return y; }
  function step() {
    var gw = [], gb = [], gc = [], gd = 0, i, k, T = TARGETS[tgt].f;
    for (i = 0; i < H; i++) { gw.push(0); gb.push(0); gc.push(0); }
    mse = 0;
    for (k = 0; k < N; k++) {
      var x = XS[k], err = predict(x) - T(x); mse += err * err / N;
      var e2 = 2 * err / N; gd += e2;
      for (i = 0; i < H; i++) { var z = net.w[i] * x + net.b[i]; if (z > 0) { gc[i] += e2 * z; gw[i] += e2 * net.c[i] * x; gb[i] += e2 * net.c[i]; } }
    }
    var g = []; for (i = 0; i < H; i++) { g.push(gw[i]); g.push(gb[i]); g.push(gc[i]); } g.push(gd);
    net.t++; var b1 = 0.9, b2 = 0.999, lr = lrS.get(), th = [];
    for (i = 0; i < H; i++) { th.push(net.w[i]); th.push(net.b[i]); th.push(net.c[i]); } th.push(net.d);
    for (i = 0; i < g.length; i++) {
      net.m[i] = b1 * net.m[i] + (1 - b1) * g[i]; net.v[i] = b2 * net.v[i] + (1 - b2) * g[i] * g[i];
      th[i] -= lr * (net.m[i] / (1 - Math.pow(b1, net.t))) / (Math.sqrt(net.v[i] / (1 - Math.pow(b2, net.t))) + 1e-8);
    }
    for (i = 0; i < H; i++) { net.w[i] = th[3 * i]; net.b[i] = th[3 * i + 1]; net.c[i] = th[3 * i + 2]; } net.d = th[3 * H];
    steps++;
  }
  function setRun(r) { running = r; runBtn.textContent = r ? 'Pause' : 'Train'; cancelAnimationFrame(raf); if (r && visible) raf = requestAnimationFrame(loop); }
  function loop() { if (!running) return; for (var i = 0; i < 8; i++) step(); c.redraw(); if (steps > 60000) { setRun(false); return; } raf = requestAnimationFrame(loop); }
  if (window.IntersectionObserver) new IntersectionObserver(function (es) { visible = es[0].isIntersecting; if (!visible) cancelAnimationFrame(raf); else if (running) raf = requestAnimationFrame(loop); }).observe(root);

  function draw() {
    if (!c.w || !net) return;
    var col = DL.colors(), T = TARGETS[tgt].f, i, ys = [], ts = [], xs = DL.linspace(-1, 1, 200), cur = 0;
    P.clear(); P.grid(); P.axes();
    if (tog.get()) for (i = 0; i < H; i++) {
      (function (i) { P.line(xs, xs.map(function (x) { var z = net.w[i] * x + net.b[i]; return z > 0 ? net.c[i] * z + net.d : net.d; }), col.muted, 1); })(i);
    }
    P.line(xs, xs.map(T), col.muted, 2.5, [6, 4]);
    P.line(xs, xs.map(predict), col.accent, 2.8);
    var kx = [], ky = [];
    for (i = 0; i < H; i++) if (Math.abs(net.w[i]) > 1e-9) { var k = -net.b[i] / net.w[i]; if (k >= -1 && k <= 1) { kx.push(k); ky.push(predict(k)); } }
    P.points(kx, ky, col.blue, 3.5);
    var m2 = 0; for (i = 0; i < N; i++) { var e = predict(XS[i]) - T(XS[i]); m2 += e * e / N; }
    say('hidden units <b>' + H + '</b> &nbsp; parameters <b>' + (3 * H + 1) + '</b> &nbsp; Adam steps <b>' + steps + '</b> &nbsp; train MSE <b>' + (isFinite(m2) ? m2.toExponential(2) : 'nan') + '</b> &nbsp; visible kinks <b>' + kx.length + '</b> (at most ' + (kx.length + 1) + ' linear pieces).');
  }
  c.draw = draw;
  init();
});
