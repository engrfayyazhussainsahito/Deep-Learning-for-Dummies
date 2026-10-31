DL.register('week15-bandit', function (root) {
  var K = 10, ctrl = DL.controls(root);
  var strat = DL.select(ctrl, { label: 'Strategy', value: 'eps', options: [{ value: 'eps', label: 'ε-greedy' }, { value: 'ucb', label: 'UCB1 (c below)' }, { value: 'ts', label: 'Thompson sampling' }] }, reset);
  var eps = DL.slider(ctrl, { label: 'ε', min: 0, max: 1, step: 0.01, value: 0.1 }, null);
  var cc = DL.slider(ctrl, { label: 'UCB c', min: 0, max: 4, step: 0.1, value: 2 }, null);
  var row = DL.row(root);
  var rand, q, N, Qh, S2, hist, tot, nstep, best, regret, timer = null, vis = true;
  DL.button(row, 'Pull ×1', function () { run(1); });
  DL.button(row, '×100', function () { run(100); });
  var play = DL.button(row, 'Play', function () { if (timer) stop(); else start(); });
  DL.button(row, 'New problem', function () { reset(true); });
  var c = DL.canvas(root, { aspect: 0.55 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Bars show the estimated value of each arm (with a marker at the true mean). Numbers on top are pull counts. Try ε = 0: the learner often stays on a mediocre arm. Try UCB with c = 0, then c = 4. Total regret is the reward you gave up compared with always pulling the best arm; a good strategy makes it grow slowly.' }));
  var seed = 1;
  function reset(newp) {
    stop(); if (newp === true) seed++;
    rand = DL.rng(seed * 101); var g = gauss; q = []; N = []; Qh = []; S2 = [];
    for (var a = 0; a < K; a++) { q.push(g()); N.push(0); Qh.push(0); S2.push(0); }
    best = 0; for (a = 1; a < K; a++) if (q[a] > q[best]) best = a;
    hist = []; tot = 0; nstep = 0; regret = 0; draw();
  }
  function gauss() { var u = 1 - rand(), v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.2832 * v); }
  function choose() {
    var a, s = strat.get();
    for (a = 0; a < K; a++) if (N[a] === 0 && s !== 'eps') return a;   // try each arm once
    if (s === 'eps') {
      if (rand() < eps.get()) return Math.floor(rand() * K);
      var m = 0; for (a = 1; a < K; a++) if (Qh[a] > Qh[m] + 1e-12) m = a; return m;
    }
    if (s === 'ucb') {
      var bi = 0, bv = -1e9; for (a = 0; a < K; a++) { var u = Qh[a] + cc.get() * Math.sqrt(Math.log(nstep + 1) / N[a]); if (u > bv) { bv = u; bi = a; } } return bi;
    }
    // Thompson sampling with a Gaussian posterior (unit reward variance, N(0,1) prior on the mean)
    var bs = 0, bz = -1e9; for (a = 0; a < K; a++) { var z = (N[a] * Qh[a]) / (N[a] + 1) + gauss() / Math.sqrt(N[a] + 1); if (z > bz) { bz = z; bs = a; } } return bs;
  }
  function run(n) {
    for (var i = 0; i < n; i++) {
      var a = choose(), r = q[a] + gauss();
      N[a]++; Qh[a] += (r - Qh[a]) / N[a]; tot += r; nstep++; regret += q[best] - q[a];
    }
    draw();
  }
  function start() { play.textContent = 'Pause'; timer = setInterval(function () { if (vis) run(5); }, 120); }
  function stop() { if (timer) clearInterval(timer); timer = null; if (play) play.textContent = 'Play'; }
  if (window.IntersectionObserver) new IntersectionObserver(function (e) { vis = e[0].isIntersecting; }).observe(root);
  function draw() {
    if (!c.w) return;
    var ctx = c.ctx, col = DL.colors(), W = c.w, H = c.h, l = 40, r = 10, t = 26, b = 30, bw = (W - l - r) / K;
    ctx.clearRect(0, 0, W, H);
    var lo = -3, hi = 3, y = function (v) { return t + (hi - v) / (hi - lo) * (H - t - b); };
    ctx.strokeStyle = col.line; ctx.beginPath(); ctx.moveTo(l, y(0)); ctx.lineTo(W - r, y(0)); ctx.stroke();
    ctx.fillStyle = col.muted; ctx.font = '11px sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    [-2, 0, 2].forEach(function (v) { ctx.fillText(String(v), l - 6, y(v)); });
    for (var a = 0; a < K; a++) {
      var x = l + a * bw + bw * 0.15, w = bw * 0.7, v = Qh[a];
      ctx.fillStyle = a === best ? col.green : col.blue; ctx.globalAlpha = 0.55;
      ctx.fillRect(x, Math.min(y(v), y(0)), w, Math.abs(y(v) - y(0))); ctx.globalAlpha = 1;
      ctx.strokeStyle = col.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x - 2, y(q[a])); ctx.lineTo(x + w + 2, y(q[a])); ctx.stroke(); ctx.lineWidth = 1;
      ctx.fillStyle = col.ink; ctx.textAlign = 'center'; ctx.fillText(String(N[a]), x + w / 2, 12);
      ctx.fillStyle = col.muted; ctx.fillText('arm ' + (a + 1), x + w / 2, H - 14);
    }
    say('pulls: <b>' + nstep + '</b> &nbsp; avg reward: <b>' + DL.fmt(nstep ? tot / nstep : 0, 3) + '</b> &nbsp; optimal-arm mean: <b>' + DL.fmt(q[best], 3) + '</b> &nbsp; total regret: <b>' + DL.fmt(regret, 2) + '</b> &nbsp; best arm pulled: <b>' + (nstep ? Math.round(100 * N[best] / nstep) : 0) + '%</b> &nbsp; <span style="opacity:.7">(black line = true mean)</span>');
  }
  c.draw = draw; reset();
});
