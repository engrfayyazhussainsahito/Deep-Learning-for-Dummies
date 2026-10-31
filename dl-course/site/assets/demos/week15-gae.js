DL.register('week15-gae', function (root) {
  // A synthetic 40-step rollout with a noisy critic. Shows the weights (gamma*lambda)^l on each TD error
  // and the resulting advantage estimate at t = 0 across lambda, using fixed random draws.
  var T = 40, ctrl = DL.controls(root);
  var gam = DL.slider(ctrl, { label: 'γ', min: 0.8, max: 0.999, step: 0.001, value: 0.99 }, draw);
  var lam = DL.slider(ctrl, { label: 'λ', min: 0, max: 1, step: 0.01, value: 0.95 }, draw);
  var noise = DL.slider(ctrl, { label: 'Reward noise σ', min: 0, max: 3, step: 0.1, value: 1 }, draw);
  var cerr = DL.slider(ctrl, { label: 'Critic error', min: 0, max: 1, step: 0.05, value: 0.3 }, draw);
  var row = DL.row(root);
  DL.button(row, 'Resample rollout', function () { seed++; gen(); draw(); });
  var seed = 3, z1, z2;
  var c = DL.canvas(root, { aspect: 0.6 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Top: weight (γλ)^l given to the TD error l steps ahead. λ = 0 uses only the first; λ = 1 (with γ ≈ 1) weighs all equally, i.e. the Monte Carlo advantage. Bottom: advantage estimate at t = 0 for every λ on this one rollout, with the chosen reward noise and critic error. Resample to see how the high-λ end jumps around more.' }));
  function gen() {
    var r = DL.rng(seed * 977), i; z1 = []; z2 = [];
    for (i = 0; i <= T; i++) { z1.push(DL.randn(r)); z2.push(DL.randn(r)); }
  }
  // true V(t) for constant reward 1 and horizon T: sum_{k<T-t} gamma^k
  function trueV(t, g) { var s = 0, p = 1; for (var k = 0; k < T - t; k++) { s += p; p *= g; } return s; }
  function estimate(l, g) {
    var V = [], r = [], t, d = [];
    for (t = 0; t <= T; t++) V.push(t === T ? 0 : trueV(t, g) + cerr.get() * 2 * z2[t]);
    for (t = 0; t < T; t++) r.push(1 + noise.get() * z1[t]);
    for (t = 0; t < T; t++) d.push(r[t] + g * V[t + 1] - V[t]);
    var A = 0, w = 1;
    for (t = 0; t < T; t++) { A += w * d[t]; w *= g * l; }
    return A;
  }
  function draw() {
    if (!c.w) return;
    var ctx = c.ctx, col = DL.colors(), g = gam.get(), l = lam.get(), W = c.w, H = c.h, i;
    ctx.clearRect(0, 0, W, H);
    // top panel: weights
    var P1 = DL.plot({ ctx: ctx, w: W, h: H / 2 }, { xmin: 0, xmax: T, ymin: 0, ymax: 1.05, xlabel: 'steps ahead l', ylabel: 'weight (γλ)^l', pad: { l: 52, r: 12, t: 12, b: 34 } });
    P1.grid(); P1.axes();
    var xs = [], ys = []; for (i = 0; i < T; i++) { xs.push(i); ys.push(Math.pow(g * l, i)); }
    P1.line(xs, ys, col.accent, 2.5); P1.points(xs, ys, col.accent, 2.5);
    // bottom panel: estimate vs lambda
    ctx.save(); ctx.translate(0, H / 2);
    var ls = [], es = [], lo = 1e9, hi = -1e9;
    for (i = 0; i <= 50; i++) { var L = i / 50; ls.push(L); var e = estimate(L, g); es.push(e); lo = Math.min(lo, e); hi = Math.max(hi, e); }
    var span = Math.max(1e-6, hi - lo);
    var P2 = DL.plot({ ctx: ctx, w: W, h: H / 2 }, { xmin: 0, xmax: 1, ymin: lo - 0.1 * span, ymax: hi + 0.1 * span, xlabel: 'λ', ylabel: 'Â₀ estimate', pad: { l: 52, r: 12, t: 12, b: 34 } });
    P2.grid(); P2.axes(); P2.line(ls, es, col.blue, 2.5); P2.points([l], [estimate(l, g)], col.red, 5);
    ctx.restore();
    var eff = (g * l) < 1 ? 1 / (1 - g * l) : Infinity;
    say('γλ = <b>' + DL.fmt(g * l, 4) + '</b> &nbsp; effective horizon 1/(1−γλ) ≈ <b>' + (isFinite(eff) ? DL.fmt(Math.min(eff, 1e6), 1) : '∞') + '</b> steps &nbsp; Â₀(λ) = <b>' + DL.fmt(estimate(l, g), 2) + '</b> &nbsp; <span style="opacity:.7">(truth for a perfect critic: 0)</span>');
  }
  c.draw = draw; gen();
});
