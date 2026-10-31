DL.register('week17-adversarial-playground', function (root) {
  // 2D two-class problem. A logistic model p(y=1|x)=sigmoid(w.x+b) is fit by gradient descent.
  // The user picks an attack, eps and steps; clicking the canvas moves the test point.
  var ctrl = DL.controls(root);
  var attack = DL.select(ctrl, { label: 'Attack', value: 'fgsm', options: [
    { value: 'fgsm', label: 'FGSM (L-inf, 1 step)' }, { value: 'pgd', label: 'PGD (L-inf, iterated)' }, { value: 'l2', label: 'L2 gradient step' }] }, draw);
  var eps = DL.slider(ctrl, { label: 'Budget ε', min: 0, max: 2, step: 0.02, value: 0.6, format: function (v) { return DL.fmt(v, 2); } }, draw);
  var steps = DL.slider(ctrl, { label: 'PGD steps', min: 1, max: 20, step: 1, value: 8, format: function (v) { return String(Math.round(v)); } }, draw);
  var sep = DL.slider(ctrl, { label: 'Class separation', min: 0.8, max: 3, step: 0.1, value: 1.8, format: function (v) { return DL.fmt(v, 1); } }, refit);
  var row = DL.row(root);
  DL.button(row, 'New data', function () { seed++; refit(); });
  DL.button(row, 'Adversarial training', function () { advTrain(); }, 'secondary');
  DL.button(row, 'Reset model', function () { refit(); });
  var c = DL.canvas(root, { aspect: 0.62 });
  var P = DL.plot(c, { xmin: -4, xmax: 4, ymin: -3, ymax: 3, xlabel: 'x₁', ylabel: 'x₂' });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Click the plot to move the test point (its reference class is the model clean prediction). The gradient of the loss with respect to the input is always parallel to w, so FGSM moves along sign(w): a diagonal step that is longer in L2 than ε. Raise ε until the label flips; then press "Adversarial training" and see how w shrinks toward the robust direction and how robust accuracy changes.' }));

  var seed = 1, X = [], Y = [], w = [0, 0], b = 0, pt = { x: 1.2, y: 0.4, y0: 1 };
  function data() {
    var r = DL.rng(seed), g = function () { return DL.randn(r); }, m = sep.get() / 2;
    X = []; Y = [];
    for (var i = 0; i < 120; i++) {
      var yy = i % 2;
      var cx = yy ? m : -m, cy = yy ? 0.4 : -0.4;
      X.push([cx + g() * 0.9, cy + g() * 0.9]); Y.push(yy);
    }
  }
  function fit(Xs, iters) {
    for (var t = 0; t < iters; t++) {
      var gw = [0, 0], gb = 0;
      for (var i = 0; i < Xs.length; i++) {
        var p = DL.sigmoid(w[0] * Xs[i][0] + w[1] * Xs[i][1] + b), e = p - Y[i % Y.length];
        gw[0] += e * Xs[i][0]; gw[1] += e * Xs[i][1]; gb += e;
      }
      var n = Xs.length;
      w[0] -= 0.5 * gw[0] / n; w[1] -= 0.5 * gw[1] / n; b -= 0.5 * gb / n;
    }
  }
  function refit() { data(); w = [0, 0]; b = 0; fit(X, 400); pt.y0 = pt.x > 0 ? 1 : 0; draw(); }
  function adv(x, y, e, kind, k) {
    // gradient of CE loss wrt input: (p - y) * w. Sign handled; zero-gradient guarded.
    var xa = [x[0], x[1]], n = kind === 'pgd' ? k : 1, alpha = kind === 'pgd' ? 2.5 * e / n : e;
    for (var i = 0; i < n; i++) {
      var p = DL.sigmoid(w[0] * xa[0] + w[1] * xa[1] + b), s = (p - y);
      var g = [s * w[0], s * w[1]], gn = Math.sqrt(g[0] * g[0] + g[1] * g[1]) + 1e-12;
      if (kind === 'l2') { xa[0] += e * g[0] / gn; xa[1] += e * g[1] / gn; }
      else {
        xa[0] += alpha * Math.sign(g[0]); xa[1] += alpha * Math.sign(g[1]);
        xa[0] = Math.min(Math.max(xa[0], x[0] - e), x[0] + e); xa[1] = Math.min(Math.max(xa[1], x[1] - e), x[1] + e);
      }
    }
    return xa;
  }
  function advTrain() {
    var e = eps.get(), kind = attack.get(), k = Math.round(steps.get());
    for (var round = 0; round < 60; round++) {
      var Xa = X.map(function (x, i) { return adv(x, Y[i], e, kind, k); });
      var tmpY = Y; fit(Xa, 5); Y = tmpY;
    }
    draw();
  }
  function acc(Xs) { var c0 = 0; for (var i = 0; i < Xs.length; i++) c0 += ((w[0] * Xs[i][0] + w[1] * Xs[i][1] + b > 0) ? 1 : 0) === Y[i] ? 1 : 0; return c0 / Xs.length; }
  DL.drag(c.canvas, { down: function (px, py) { move(px, py); }, move: function (px, py) { move(px, py); } });
  function move(px, py) { pt.x = P.dx(px); pt.y = P.dy(py); pt.y0 = (w[0] * pt.x + w[1] * pt.y + b) > 0 ? 1 : 0; draw(); }
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), e = eps.get(), kind = attack.get(), k = Math.round(steps.get());
    P.clear();
    P.heat(function (x, y) { return DL.sigmoid(w[0] * x + w[1] * y + b); }, function (t) { return DL.cmap.diverge(t); }, 8);
    P.grid(); P.axes();
    for (var i = 0; i < X.length; i++) P.points([X[i][0]], [X[i][1]], Y[i] ? col.red : col.blue, 2.8);
    // decision boundary
    if (Math.abs(w[1]) > 1e-6) P.line([-4, 4], [(-b + 4 * w[0]) / w[1], (-b - 4 * w[0]) / w[1]], col.ink, 2);
    else P.vline(-b / (w[0] || 1), col.ink, []);
    var x0 = [pt.x, pt.y], ytrue = pt.y0, xa = adv(x0, ytrue, e, kind, k);
    var ctx = c.ctx;
    ctx.save(); ctx.strokeStyle = col.muted; ctx.setLineDash([4, 3]);
    var q = P.px(x0[0]), r = P.py(x0[1]), s = P.px(x0[0] + e) - q;
    if (kind === 'l2') { ctx.beginPath(); ctx.arc(q, r, Math.abs(s), 0, 6.2832); ctx.stroke(); } else ctx.strokeRect(q - Math.abs(s), r - Math.abs(s), 2 * Math.abs(s), 2 * Math.abs(s));
    ctx.restore();
    P.line([x0[0], xa[0]], [x0[1], xa[1]], col.amber, 2);
    P.points([x0[0]], [x0[1]], ytrue ? col.red : col.blue, 6);
    P.points([xa[0]], [xa[1]], col.amber, 5);
    var sc0 = w[0] * x0[0] + w[1] * x0[1] + b, sc1 = w[0] * xa[0] + w[1] * xa[1] + b;
    var flipped = (sc1 > 0 ? 1 : 0) !== ytrue;
    var Xa = X.map(function (x, i) { return adv(x, Y[i], e, kind, k); });
    var wn = Math.sqrt(w[0] * w[0] + w[1] * w[1]);
    say('clean acc <b>' + DL.fmt(100 * acc(X), 1) + '%</b> &nbsp; robust acc (' + kind + ', ε=' + DL.fmt(e, 2) + ') <b>' + DL.fmt(100 * acc(Xa), 1) + '%</b> &nbsp; |w| = ' + DL.fmt(wn, 2) + ' &nbsp; logit: ' + DL.fmt(sc0, 2) + ' → ' + DL.fmt(sc1, 2) + ' &nbsp; <b>' + (flipped ? 'prediction flipped' : 'prediction unchanged') + '</b>. Linear logit change at most ε·|w|₁ = ' + DL.fmt(e * (Math.abs(w[0]) + Math.abs(w[1])), 2));
  }
  c.draw = draw;
  refit();
});
