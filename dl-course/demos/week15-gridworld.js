DL.register('week15-gridworld', function (root) {
  // 5x5 gridworld: start top-left, goal bottom-right (+1), pit (-1), one wall. Slip probability slider.
  var H = 5, W = 5, S = H * W, GOAL = 24, PIT = 18, WALL = { 7: 1, 12: 1 }, START = 0;
  var MV = [[-1, 0], [0, 1], [1, 0], [0, -1]], ARROW = ['↑', '→', '↓', '←'];
  var ctrl = DL.controls(root);
  var mode = DL.select(ctrl, { label: 'Algorithm', value: 'vi', options: [{ value: 'vi', label: 'Value iteration (known model)' }, { value: 'ql', label: 'Q-learning (samples only)' }] }, reset);
  var gam = DL.slider(ctrl, { label: 'Discount γ', min: 0.5, max: 0.99, step: 0.01, value: 0.9 }, reset);
  var slip = DL.slider(ctrl, { label: 'Slip prob.', min: 0, max: 0.5, step: 0.05, value: 0.2 }, reset);
  var eps = DL.slider(ctrl, { label: 'ε (Q-learning)', min: 0, max: 1, step: 0.05, value: 0.2 }, function () { draw(); });
  var alpha = DL.slider(ctrl, { label: 'α (Q-learning)', min: 0.02, max: 1, step: 0.02, value: 0.3 }, function () { draw(); });
  var row = DL.row(root);
  var rand = DL.rng(7);
  var V, Q, sweeps, episodes, steps, agent, timer = null, lastDelta;
  DL.button(row, 'Step / sweep', function () { advance(1); });
  DL.button(row, '×10', function () { advance(10); });
  DL.button(row, '×100', function () { advance(100); });
  var playBtn = DL.button(row, 'Play', function () { if (timer) stop(); else play(); });
  DL.button(row, 'Reset', reset);
  var c = DL.canvas(root, { aspect: 0.62 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Value iteration needs the model and sweeps every state; watch values spread backwards from the goal. Q-learning only sees sampled steps (the dot is the agent): values appear along paths it has actually travelled. Try a high slip probability: the greedy path moves away from the pit. Try ε = 0 in Q-learning and see how exploration affects what is learned.' }));

  function reset() {
    stop(); rand = DL.rng(7);
    V = []; Q = []; var s;
    for (s = 0; s < S; s++) { V.push(0); Q.push([0, 0, 0, 0]); }
    sweeps = 0; episodes = 0; steps = 0; agent = START; lastDelta = NaN; draw();
  }
  function mv(s, a) {
    var r = Math.floor(s / W), cc = s % W, r2 = r + MV[a][0], c2 = cc + MV[a][1];
    if (r2 < 0 || r2 >= H || c2 < 0 || c2 >= W) return s;
    var n = r2 * W + c2; return WALL[n] ? s : n;
  }
  function rew(s2) { return s2 === GOAL ? 1 : (s2 === PIT ? -1 : -0.04); }
  function outcomes(s, a) {
    var p = slip.get();
    return [[a, 1 - p], [(a + 1) % 4, p / 2], [(a + 3) % 4, p / 2]].map(function (x) { return [mv(s, x[0]), x[1]]; });
  }
  function term(s) { return s === GOAL || s === PIT; }
  function qvi(s, a) {
    var g = gam.get(), t = 0; outcomes(s, a).forEach(function (o) { t += o[1] * (rew(o[0]) + g * (term(o[0]) ? 0 : V[o[0]])); }); return t;
  }
  function sweep() {
    var d = 0, nv = V.slice();
    for (var s = 0; s < S; s++) {
      if (term(s) || WALL[s]) continue;
      var b = -1e9; for (var a = 0; a < 4; a++) b = Math.max(b, qvi(s, a));
      d = Math.max(d, Math.abs(b - V[s])); nv[s] = b;
    }
    V = nv; sweeps++; lastDelta = d;
  }
  function sample(s, a) {
    var p = slip.get(), u = rand(), a2 = a;
    if (u < p / 2) a2 = (a + 1) % 4; else if (u < p) a2 = (a + 3) % 4;
    return mv(s, a2);
  }
  function qstep() {
    var s = agent, a, g = gam.get();
    if (rand() < eps.get()) a = Math.floor(rand() * 4); else { a = 0; for (var k = 1; k < 4; k++) if (Q[s][k] > Q[s][a] + 1e-12) a = k; }
    var s2 = sample(s, a), r = rew(s2), tgt = r + (term(s2) ? 0 : g * Math.max.apply(null, Q[s2]));
    Q[s][a] += alpha.get() * (tgt - Q[s][a]); steps++;
    agent = term(s2) ? START : s2; if (term(s2)) episodes++;
  }
  function advance(n) {
    for (var i = 0; i < n; i++) { if (mode.get() === 'vi') sweep(); else qstep(); }
    draw();
  }
  function play() { playBtn.textContent = 'Pause'; timer = setInterval(function () { advance(mode.get() === 'vi' ? 1 : 5); }, 140); }
  function stop() { if (timer) clearInterval(timer); timer = null; if (playBtn) playBtn.textContent = 'Play'; }
  function vOf(s) { return mode.get() === 'vi' ? V[s] : Math.max.apply(null, Q[s]); }
  function best(s) {
    var q = []; for (var a = 0; a < 4; a++) q.push(mode.get() === 'vi' ? qvi(s, a) : Q[s][a]);
    var m = 0; for (var k = 1; k < 4; k++) if (q[k] > q[m] + 1e-9) m = k;
    return Math.abs(q[m]) < 1e-12 && mode.get() === 'ql' ? -1 : m;
  }
  function draw() {
    if (!c.w) return;
    var ctx = c.ctx, col = DL.colors(), cw = Math.min(c.w / W, c.h / H), ox = (c.w - cw * W) / 2, oy = (c.h - cw * H) / 2;
    ctx.clearRect(0, 0, c.w, c.h);
    for (var s = 0; s < S; s++) {
      var r = Math.floor(s / W), cc = s % W, x = ox + cc * cw, y = oy + r * cw;
      if (WALL[s]) { ctx.fillStyle = col.muted; ctx.fillRect(x + 1, y + 1, cw - 2, cw - 2); continue; }
      var v = vOf(s), t = Math.max(-1, Math.min(1, v));
      ctx.fillStyle = t >= 0 ? col.green : col.red; ctx.globalAlpha = 0.08 + 0.55 * Math.abs(t); ctx.fillRect(x + 1, y + 1, cw - 2, cw - 2); ctx.globalAlpha = 1;
      ctx.strokeStyle = col.line; ctx.strokeRect(x + 1, y + 1, cw - 2, cw - 2);
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = col.ink;
      if (s === GOAL) { ctx.font = 'bold ' + Math.round(cw * 0.22) + 'px sans-serif'; ctx.fillText('GOAL +1', x + cw / 2, y + cw / 2); continue; }
      if (s === PIT) { ctx.font = 'bold ' + Math.round(cw * 0.22) + 'px sans-serif'; ctx.fillText('PIT −1', x + cw / 2, y + cw / 2); continue; }
      var b = best(s);
      if (b >= 0) { ctx.font = Math.round(cw * 0.42) + 'px sans-serif'; ctx.fillText(ARROW[b], x + cw / 2, y + cw * 0.42); }
      ctx.font = Math.round(cw * 0.16) + 'px monospace'; ctx.fillStyle = col.muted; ctx.fillText(DL.fmt(v, 2), x + cw / 2, y + cw * 0.84);
      if (s === START) { ctx.font = Math.round(cw * 0.14) + 'px sans-serif'; ctx.textAlign = 'left'; ctx.fillText('start', x + 4, y + 10); }
    }
    if (mode.get() === 'ql') {
      var ar = Math.floor(agent / W), ac = agent % W;
      ctx.fillStyle = col.accent; ctx.beginPath(); ctx.arc(ox + ac * cw + cw * 0.18, oy + ar * cw + cw * 0.78, cw * 0.07, 0, 6.2832); ctx.fill();
    }
    if (mode.get() === 'vi') say('sweeps: <b>' + sweeps + '</b> &nbsp; max change last sweep: <b>' + (isNaN(lastDelta) ? '-' : lastDelta.toExponential(2)) + '</b> &nbsp; V(start) = <b>' + DL.fmt(V[START], 3) + '</b>');
    else say('steps: <b>' + steps + '</b> &nbsp; episodes finished: <b>' + episodes + '</b> &nbsp; max<sub>a</sub> Q(start,a) = <b>' + DL.fmt(Math.max.apply(null, Q[START]), 3) + '</b> &nbsp; (arrows show greedy action where Q is non-zero)');
  }
  c.draw = draw;
  reset();
});
