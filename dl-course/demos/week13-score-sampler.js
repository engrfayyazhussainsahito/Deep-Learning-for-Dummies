DL.register('week13-score-sampler', function (root) {
  var T = 1000;
  var ctrl = DL.controls(root);
  var tsl = DL.slider(ctrl, { label: 'noise level t', min: 1, max: 1000, step: 1, value: 300 }, draw);
  var samp = DL.select(ctrl, { label: 'Sampler', value: 'ddim', options: [{ value: 'ddpm', label: 'DDPM (stochastic, η = 1)' }, { value: 'ddim', label: 'DDIM (deterministic, η = 0)' }, { value: 'euler', label: 'naive Euler on the data (no noise)' }] }, resetP);
  var nst = DL.slider(ctrl, { label: 'steps', min: 5, max: 200, step: 1, value: 40 }, resetP);
  var field = DL.toggle(ctrl, 'show score arrows', true, draw);
  var row = DL.row(root);
  DL.button(row, 'run reverse process', function () { resetP(); run(); });
  DL.button(row, 'reset', resetP);
  var c = DL.canvas(root, { aspect: 0.62 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'The data distribution is a mixture of 3 Gaussians; the score of every noised marginal is known in closed form, so no network is needed. Move t: at large t the arrows point to the global center, at small t they snap to the nearest mode. "run" starts 300 particles from N(0, I) at t = 1000 and follows the learned-score update. Try 5-10 steps with each sampler: DDIM stays near the modes much better than ancestral DDPM at few steps. "Naive Euler" uses ε-prediction to jump straight to the x₀ estimate and shows what happens if you forget the noise term (samples collapse to mode centers).' }));
  var MU = [[-1.6, -0.9], [1.6, -0.9], [0, 1.6]], S0 = 0.35;
  var ABS = new Array(T + 1); (function () { var s = 1; ABS[0] = 1; for (var i = 0; i < T; i++) { s *= 1 - (1e-4 + (0.02 - 1e-4) * i / (T - 1)); ABS[i + 1] = s; } })();
  function eps(x, y, k) { // exact eps*(x_t,t) = -sqrt(1-ab) * score
    var a = ABS[k], sa = Math.sqrt(a), v = a * S0 * S0 + 1 - a, lp = [], mx = -1e30, i;
    for (i = 0; i < 3; i++) { var dx = x - sa * MU[i][0], dy = y - sa * MU[i][1]; lp.push(-(dx * dx + dy * dy) / (2 * v)); if (lp[i] > mx) mx = lp[i]; }
    var Z = 0, sx = 0, sy = 0;
    for (i = 0; i < 3; i++) { var w = Math.exp(lp[i] - mx); Z += w; sx += w * (x - sa * MU[i][0]); sy += w * (y - sa * MU[i][1]); }
    var sc = Math.sqrt(1 - a) / v / Z; return [sc * sx, sc * sy];
  }
  var NP = 300, rand = DL.rng(5), rn = function () { return DL.randn(rand); }, pts = [], cur = T, anim = null, ks = [];
  function resetP() {
    if (anim) cancelAnimationFrame(anim); anim = null;
    pts = []; for (var i = 0; i < NP; i++) pts.push([rn(), rn()]);
    var n = Math.round(nst.get()); ks = []; for (var j = 0; j <= n; j++) ks.push(Math.round(T - (T - 0) * j / n));
    cur = 0; // index into ks
    draw();
  }
  function stepOnce() {
    if (cur >= ks.length - 1) return false;
    var a = ks[cur], b = ks[cur + 1], A = ABS[a], B = ABS[b], mode = samp.get();
    for (var i = 0; i < NP; i++) {
      var p = pts[i], e = eps(p[0], p[1], a), x0x = (p[0] - Math.sqrt(1 - A) * e[0]) / Math.sqrt(A), x0y = (p[1] - Math.sqrt(1 - A) * e[1]) / Math.sqrt(A);
      if (mode === 'euler') { p[0] = Math.sqrt(B) * x0x + Math.sqrt(1 - B) * 0; p[1] = Math.sqrt(B) * x0y; continue; }
      var eta = mode === 'ddpm' ? 1 : 0;
      var sig = eta * Math.sqrt((1 - B) / (1 - A)) * Math.sqrt(Math.max(1 - A / B, 0));
      var dir = Math.sqrt(Math.max(1 - B - sig * sig, 0));
      p[0] = Math.sqrt(B) * x0x + dir * e[0] + sig * rn(); p[1] = Math.sqrt(B) * x0y + dir * e[1] + sig * rn();
    }
    cur++; return true;
  }
  function run() {
    (function loop() {
      if (!root.isConnected) { anim = null; return; }
      var r = c.canvas.getBoundingClientRect();
      if (r.bottom > 0 && r.top < (window.innerHeight || 800)) { if (!stepOnce()) { anim = null; draw(); return; } draw(); }
      anim = setTimeout(function () { anim = requestAnimationFrame(loop); }, 60);
    })();
  }
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), P = DL.plot(c, { xmin: -3.4, xmax: 3.4, ymin: -2.8, ymax: 2.8, xlabel: 'x₁', ylabel: 'x₂' });
    P.clear(); P.grid(); P.axes();
    var t = Math.round(tsl.get()), a = ABS[t], sa = Math.sqrt(a), ctx = c.ctx;
    var v = a * S0 * S0 + 1 - a;
    P.contour(function (x, y) { var s = 0; for (var i = 0; i < 3; i++) { var dx = x - sa * MU[i][0], dy = y - sa * MU[i][1]; s += Math.exp(-(dx * dx + dy * dy) / (2 * v)); } return s / (3 * 2 * Math.PI * v) * 2; }, [0.03, 0.08, 0.2, 0.4, 0.7], col.blue);
    if (field.get()) {
      ctx.save(); ctx.strokeStyle = col.accent; ctx.fillStyle = col.accent; ctx.lineWidth = 1.4;
      for (var gx = -3; gx <= 3.01; gx += 0.6) for (var gy = -2.4; gy <= 2.41; gy += 0.6) {
        var e = eps(gx, gy, t), sc = -1 / Math.sqrt(1 - a); // score = -eps / sqrt(1-ab)
        var sx = e[0] * sc, sy = e[1] * sc, m = Math.sqrt(sx * sx + sy * sy), L = Math.min(0.5, 0.18 * m);
        if (m < 1e-9) continue;
        var ux = sx / m, uy = sy / m, x0 = P.px(gx), y0 = P.py(gy), x1 = P.px(gx + ux * L), y1 = P.py(gy + uy * L);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        var ang = Math.atan2(y1 - y0, x1 - x0); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - 6 * Math.cos(ang - 0.45), y1 - 6 * Math.sin(ang - 0.45)); ctx.lineTo(x1 - 6 * Math.cos(ang + 0.45), y1 - 6 * Math.sin(ang + 0.45)); ctx.closePath(); ctx.fill();
      }
      ctx.restore();
    }
    var xs = pts.map(function (p) { return p[0]; }), ys = pts.map(function (p) { return p[1]; });
    P.points(xs, ys, col.ink, 2.4);
    for (var i = 0; i < 3; i++) P.points([MU[i][0]], [MU[i][1]], col.red, 5);
    // fraction of particles within 2 sigma of a mode
    var near = 0; pts.forEach(function (p) { for (var k = 0; k < 3; k++) { var dx = p[0] - MU[k][0], dy = p[1] - MU[k][1]; if (dx * dx + dy * dy < 4 * S0 * S0 * 2) { near++; break; } } });
    var tcur = ks.length ? ks[Math.min(cur, ks.length - 1)] : T;
    say('score panel: t = <b>' + t + '</b>, ᾱ = <b>' + DL.fmt(a, 3) + '</b>, marginal std of each blob = <b>' + DL.fmt(Math.sqrt(v), 3) + '</b> (sa·μ shrinks, blob widens)<br>sampler: step <b>' + cur + '/' + Math.max(ks.length - 1, 0) + '</b>, current t = <b>' + tcur + '</b>, particles within 2σ of a mode: <b>' + near + '/' + NP + '</b> (dots: particles, red: true mode centers)');
  }
  c.draw = draw;
  resetP();
});
