DL.register('week13-forward-noising', function (root) {
  var T = 1000;
  var ctrl = DL.controls(root);
  var shape = DL.select(ctrl, { label: 'Data', value: 'moons', options: [{ value: 'moons', label: 'two moons' }, { value: 'ring', label: 'ring' }, { value: 'blobs', label: 'three blobs' }] }, build);
  var sched = DL.select(ctrl, { label: 'Schedule', value: 'linear', options: [{ value: 'linear', label: 'linear β (1e-4 → 0.02)' }, { value: 'cosine', label: 'cosine (s = 0.008)' }] }, draw);
  var ts = DL.slider(ctrl, { label: 't', min: 0, max: T, step: 1, value: 0 }, draw);
  var row = DL.row(root);
  DL.button(row, 'new noise ε', function () { newEps(); draw(); });
  var anim = null, playBtn = DL.button(row, 'play', function () { if (anim) stop(); else play(); });
  var c = DL.canvas(root, { aspect: 0.55 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Drag t from 0 to 1000. Each point is x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε with the SAME ε for every t, so a point glides smoothly from its data position to its noise sample. Compare schedules at t = 500: the linear schedule has already destroyed most of the structure (ᾱ ≈ 0.08), the cosine schedule has not (ᾱ ≈ 0.49). Gray ring: radius of the 2σ sphere of the final Gaussian.' }));
  var N = 400, X0 = [], EPS = [], rand = DL.rng(7), rn = function () { return DL.randn(rand); }, nrand = DL.rng(99);
  function abAt(t) {
    t = Math.max(0, Math.min(T, Math.round(t)));
    if (sched.get() === 'cosine') {
      var f = function (u) { var v = Math.cos((u / T + 0.008) / 1.008 * Math.PI / 2); return v * v; };
      return Math.min(f(t) / f(0), 0.9999);
    }
    // linear: product of (1 - beta_i) for i < t, beta_i = 1e-4 + (0.02-1e-4) i/(T-1)
    var s = 1; for (var i = 0; i < t; i++) s *= 1 - (1e-4 + (0.02 - 1e-4) * i / (T - 1));
    return s;
  }
  var cache = {};
  function ab(t) { var k = sched.get() + t; if (cache[k] == null) cache[k] = abAt(t); return cache[k]; }
  function build() {
    X0 = []; var i, a, r;
    for (i = 0; i < N; i++) {
      if (shape.get() === 'moons') {
        a = Math.PI * rand(); r = rn() * 0.06;
        if (i % 2) X0.push([(Math.cos(a) + r) * 1.0 - 0.5, (Math.sin(a) + r) * 1.0 - 0.25]);
        else X0.push([(1 - Math.cos(a) + r) - 0.5, (0.5 - Math.sin(a) + r) - 0.25]);
      } else if (shape.get() === 'ring') {
        a = 2 * Math.PI * rand(); r = 1.2 + rn() * 0.06; X0.push([r * Math.cos(a), r * Math.sin(a)]);
      } else {
        var cs = [[-1.3, -0.8], [1.3, -0.8], [0, 1.4]], k = i % 3;
        X0.push([cs[k][0] + rn() * 0.2, cs[k][1] + rn() * 0.2]);
      }
    }
    // normalise to unit overall variance (as data are in practice)
    var sx = 0, n = X0.length; for (i = 0; i < n; i++) sx += X0[i][0] * X0[i][0] + X0[i][1] * X0[i][1];
    var sc = 1 / Math.sqrt(sx / (2 * n)); for (i = 0; i < n; i++) { X0[i][0] *= sc; X0[i][1] *= sc; }
    newEps(); draw();
  }
  function newEps() { EPS = []; for (var i = 0; i < N; i++) EPS.push([rn(), rn()]); }
  function stop() { if (anim) cancelAnimationFrame(anim); anim = null; playBtn.textContent = 'play'; }
  function play() {
    if (ts.get() >= T) ts.set(0);
    playBtn.textContent = 'pause';
    (function step() {
      if (!root.isConnected) { anim = null; return; }
      var r = c.canvas.getBoundingClientRect();
      var visible = r.bottom > 0 && r.top < (window.innerHeight || 800);
      if (visible) { var v = ts.get() + 8; if (v >= T) { ts.set(T); draw(); stop(); return; } ts.set(v); draw(); }
      anim = requestAnimationFrame(step);
    })();
  }
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), t = ts.get(), a = ab(t), sa = Math.sqrt(a), sb = Math.sqrt(1 - a);
    var P = DL.plot(c, { xmin: -3.2, xmax: 3.2, ymin: -2.3, ymax: 2.3, xlabel: 'x₁', ylabel: 'x₂' });
    P.clear(); P.grid(); P.axes();
    var th = DL.linspace(0, 2 * Math.PI, 90), rx = [], ry = [];
    th.forEach(function (u) { rx.push(2 * Math.cos(u)); ry.push(2 * Math.sin(u)); });
    P.line(rx, ry, col.muted, 1, [4, 4]);
    var xs = [], ys = [];
    for (var i = 0; i < N; i++) { xs.push(sa * X0[i][0] + sb * EPS[i][0]); ys.push(sa * X0[i][1] + sb * EPS[i][1]); }
    P.points(xs, ys, col.accent, 3);
    var snr = a / Math.max(1 - a, 1e-12);
    say('t = <b>' + t + '</b> &nbsp; ᾱ_t = <b>' + DL.fmt(a, 4) + '</b> &nbsp; signal scale √ᾱ = <b>' + DL.fmt(sa, 3) + '</b> &nbsp; noise std √(1−ᾱ) = <b>' + DL.fmt(sb, 3) + '</b><br>SNR = ᾱ/(1−ᾱ) = <b>' + (snr > 1000 ? snr.toExponential(2) : DL.fmt(snr, 3)) + '</b> &nbsp; log-SNR = <b>' + DL.fmt(Math.log(Math.max(snr, 1e-12)), 2) + '</b>');
  }
  c.draw = draw;
  build();
});
