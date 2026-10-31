DL.register('week04-l1-l2', function (root) {
  var ctrl = DL.controls(root);
  var mode = DL.select(ctrl, { label: 'Penalty', value: 'l1', options: [{ value: 'l1', label: 'L1 (lasso)' }, { value: 'l2', label: 'L2 (ridge)' }, { value: 'en', label: 'Elastic net (50/50)' }] }, draw);
  var lam = DL.slider(ctrl, { label: 'Strength λ', min: 0, max: 6, step: 0.05, value: 1.5 }, draw);
  var ang = DL.slider(ctrl, { label: 'Ellipse angle (deg)', min: 0, max: 180, step: 1, value: 35 }, draw);
  var asp = DL.slider(ctrl, { label: 'Ellipse elongation', min: 1, max: 8, step: 0.1, value: 3 }, draw);
  var c = DL.canvas(root, { aspect: 0.6 });
  var say = DL.readout(root);
  var opt = [2.6, 1.3];
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Drag the red dot (the unregularized optimum). Loss = ½(w-w*)ᵀH(w-w*) + λ·penalty. Dashed curves are loss level sets, the filled shape is the constraint region the penalty implies at the solution. With L1, many optima land exactly on an axis (a zero weight); with L2 they never do.' }));
  var P;
  function Hm() { var a = ang.get() * Math.PI / 180, e = asp.get(), ca = Math.cos(a), sa = Math.sin(a); // eigenvalues 1*e and 1/e... keep scale moderate
    var l1 = e, l2 = 1;
    return [[l1 * ca * ca + l2 * sa * sa, (l1 - l2) * ca * sa], [(l1 - l2) * ca * sa, l1 * sa * sa + l2 * ca * ca]]; }
  function pen(w, m, l) { if (m === 'l1') return l * (Math.abs(w[0]) + Math.abs(w[1])); if (m === 'l2') return 0.5 * l * (w[0] * w[0] + w[1] * w[1]); return l * 0.5 * (Math.abs(w[0]) + Math.abs(w[1])) + 0.25 * l * (w[0] * w[0] + w[1] * w[1]); }
  function soft(z, t) { return z > t ? z - t : z < -t ? z + t : 0; }
  function solve(m, l) { // coordinate descent on 0.5 (w-o)ᵀH(w-o) + penalty
    var H = Hm(), w = [0, 0], it, j;
    for (it = 0; it < 300; it++) for (j = 0; j < 2; j++) {
      var o = 1 - j, g = H[j][0] * (0 - 0); // placeholder
      var r = H[j][j] * opt[j] + H[j][o] * (opt[o] - w[o]); // minimiser of smooth part wrt w_j given w_o: (H_jj o_j + H_jo (o_o - w_o)) / H_jj
      var l1 = m === 'l1' ? l : m === 'en' ? 0.5 * l : 0, l2 = m === 'l2' ? l : m === 'en' ? 0.5 * l : 0;
      w[j] = soft(r, l1) / (H[j][j] + l2);
    }
    return w;
  }
  function loss(w) { var H = Hm(), d = [w[0] - opt[0], w[1] - opt[1]]; return 0.5 * (d[0] * (H[0][0] * d[0] + H[0][1] * d[1]) + d[1] * (H[1][0] * d[0] + H[1][1] * d[1])); }
  function draw() {
    var col = DL.colors(), m = mode.get(), l = lam.get();
    P = DL.plot(c, { xmin: -4, xmax: 4, ymin: -3, ymax: 3, xlabel: 'w₁', ylabel: 'w₂' });
    P.clear(); P.grid(); P.axes();
    var w = solve(m, l), lw = loss(w), pw = pen(w, m, 1);
    // constraint region: penalty(w)/l <= pen(w*)
    var ctx = c.ctx, pts = [], i;
    var rad = (m === 'l1') ? Math.abs(w[0]) + Math.abs(w[1]) : (m === 'l2') ? Math.hypot(w[0], w[1]) : 0;
    ctx.save(); ctx.beginPath();
    var N = 200;
    for (i = 0; i <= N; i++) { var t = 2 * Math.PI * i / N, dir = [Math.cos(t), Math.sin(t)], lo = 0, hi = 20;
      for (var b = 0; b < 40; b++) { var mid = (lo + hi) / 2; if (pen([mid * dir[0], mid * dir[1]], m, 1) <= pw) lo = mid; else hi = mid; }
      var x = P.px(lo * dir[0]), y = P.py(lo * dir[1]); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
    ctx.closePath(); ctx.fillStyle = col.accent + '33'; ctx.fill(); ctx.strokeStyle = col.accent; ctx.lineWidth = 2; ctx.stroke(); ctx.restore();
    var levels = [0.25, 0.8, 1.8, 3.5, 6].map(function (v) { return v * 0.5; }).concat([lw]);
    P.contour(function (x, y) { return loss([x, y]); }, levels, col.blue);
    ctx.fillStyle = col.red; ctx.beginPath(); ctx.arc(P.px(opt[0]), P.py(opt[1]), 7, 0, 7); ctx.fill();
    ctx.fillStyle = col.green; ctx.beginPath(); ctx.arc(P.px(w[0]), P.py(w[1]), 6, 0, 7); ctx.fill();
    P.text('w*', opt[0] + 0.15, opt[1] + 0.25, col.red); P.text('solution', w[0] + 0.15, w[1] - 0.3, col.green);
    say('w* = (' + DL.fmt(opt[0], 2) + ', ' + DL.fmt(opt[1], 2) + ') &nbsp; regularized w = (<b>' + DL.fmt(w[0], 3) + '</b>, <b>' + DL.fmt(w[1], 3) + '</b>) &nbsp; ' + ((Math.abs(w[0]) < 1e-9 || Math.abs(w[1]) < 1e-9) ? '<b>exact zero weight (sparse)</b>' : 'no exact zeros') + ' &nbsp; data loss ' + DL.fmt(lw, 3));
  }
  c.draw = draw;
  DL.drag(c.canvas, { down: mv, move: mv });
  function mv(px, py) { if (!P) return; opt = [DL.clamp(P.dx(px), -3.8, 3.8), DL.clamp(P.dy(py), -2.8, 2.8)]; draw(); }
});
