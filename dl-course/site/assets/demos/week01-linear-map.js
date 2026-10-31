DL.register('week01-linear-map', function (root) {
  var presets = {
    custom: null,
    identity: [1, 0, 0, 1],
    rotate: [Math.cos(0.6), -Math.sin(0.6), Math.sin(0.6), Math.cos(0.6)],
    scale: [1.6, 0, 0, 0.5],
    shear: [1, 0.9, 0, 1],
    reflect: [1, 0, 0, -1],
    sym: [1.5, 0.8, 0.8, 0.9],
    singular: [1, 0.5, 2, 1],
    rotscale: [0.9, -0.6, 0.6, 0.9]
  };
  var ctrl = DL.controls(root);
  var sel = DL.select(ctrl, {
    label: 'Preset', value: 'shear',
    options: [['custom', 'custom (use sliders)'], ['identity', 'identity'], ['rotate', 'rotation 0.6 rad'], ['scale', 'axis scaling diag(1.6, 0.5)'],
      ['shear', 'shear'], ['reflect', 'reflection'], ['sym', 'symmetric'], ['singular', 'singular (rank 1)'], ['rotscale', 'spiral (complex eigenvalues)']]
      .map(function (o) { return { value: o[0], label: o[1] }; })
  }, function (v) {
    if (presets[v]) { setM(presets[v]); }
    draw();
  });
  var row = DL.row(root);
  function sl(label, i) {
    return DL.slider(row, { label: label, min: -2.5, max: 2.5, step: 0.05, value: 0, format: function (v) { return (+v).toFixed(2); } }, function () { sel.set('custom'); draw(); });
  }
  var S = [sl('a (M11)'), sl('b (M12)'), sl('c (M21)'), sl('d (M22)')];
  var tg = DL.row(root);
  var showCircle = DL.toggle(tg, 'unit circle / ellipse', true, draw);
  var showSVD = DL.toggle(tg, 'singular directions', true, draw);
  var showEig = DL.toggle(tg, 'eigenvectors', true, draw);
  var showGrid = DL.toggle(tg, 'deformed grid', true, draw);
  function setM(m) { for (var i = 0; i < 4; i++) S[i].set(m[i]); }
  setM(presets.shear);
  var c = DL.canvas(root, { aspect: 0.62 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Try: rotation (det = 1, all singular values 1), the singular preset (the whole plane collapses onto a line), the symmetric preset (eigenvectors = singular directions, orthogonal) and the shear (only one real eigen-direction). The ellipse axes are always the left singular vectors scaled by the singular values.' }));

  function getM() { return [S[0].get(), S[1].get(), S[2].get(), S[3].get()]; }
  // SVD of a 2x2 matrix: returns {s1, s2, v1, v2, u1, u2}
  function svd(M) {
    var a = M[0], b = M[1], cc = M[2], d = M[3];
    var p = a * a + cc * cc, q = a * b + cc * d, r = b * b + d * d; // M^T M = [[p,q],[q,r]]
    var tr = p + r, det = p * r - q * q;
    var disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    var l1 = tr / 2 + disc, l2 = Math.max(0, tr / 2 - disc);
    var v1;
    if (Math.abs(q) > 1e-12) v1 = [l1 - r, q]; else v1 = p >= r ? [1, 0] : [0, 1];
    var n = Math.hypot(v1[0], v1[1]) || 1; v1 = [v1[0] / n, v1[1] / n];
    var v2 = [-v1[1], v1[0]];
    var s1 = Math.sqrt(l1), s2 = Math.sqrt(l2);
    function app(v) { return [a * v[0] + b * v[1], cc * v[0] + d * v[1]]; }
    var u1 = app(v1), u2 = app(v2);
    if (s1 > 1e-9) u1 = [u1[0] / s1, u1[1] / s1]; else u1 = [1, 0];
    if (s2 > 1e-9) u2 = [u2[0] / s2, u2[1] / s2]; else u2 = [-u1[1], u1[0]];
    return { s1: s1, s2: s2, v1: v1, v2: v2, u1: u1, u2: u2 };
  }
  function eig(M) {
    var a = M[0], b = M[1], cc = M[2], d = M[3];
    var tr = a + d, det = a * d - b * cc, disc = tr * tr / 4 - det;
    if (disc < -1e-12) return { re: tr / 2, im: Math.sqrt(-disc), vecs: [] };
    var s = Math.sqrt(Math.max(0, disc)), out = [];
    [tr / 2 + s, tr / 2 - s].forEach(function (lam) {
      var v;
      if (Math.abs(b) > 1e-9) v = [b, lam - a];
      else if (Math.abs(cc) > 1e-9) v = [lam - d, cc];
      else v = Math.abs(lam - a) < 1e-9 ? [1, 0] : [0, 1];
      var n = Math.hypot(v[0], v[1]) || 1;
      out.push({ lam: lam, v: [v[0] / n, v[1] / n] });
    });
    if (disc < 1e-12 && Math.abs(b) < 1e-9 && Math.abs(cc) < 1e-9 && Math.abs(a - d) < 1e-9) out = [{ lam: a, v: [1, 0] }, { lam: a, v: [0, 1] }];
    else if (disc < 1e-12) out = [out[0]];
    return { re: tr / 2, im: 0, vecs: out };
  }
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), ctx = c.ctx, M = getM();
    var R = 3.2, xr = R * c.w / c.h;
    var P = DL.plot(c, { xmin: -xr, xmax: xr, ymin: -R, ymax: R, pad: { l: 8, r: 8, t: 8, b: 8 } });
    function T(x, y) { return [M[0] * x + M[1] * y, M[2] * x + M[3] * y]; }
    function seg(x0, y0, x1, y1, color, w, dash) {
      ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = w; if (dash) ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(P.px(x0), P.py(y0)); ctx.lineTo(P.px(x1), P.py(y1)); ctx.stroke(); ctx.restore();
    }
    function arrow(x, y, color, w) {
      var X = P.px(x), Y = P.py(y), X0 = P.px(0), Y0 = P.py(0), ang = Math.atan2(Y - Y0, X - X0);
      ctx.save(); ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = w || 2.5;
      ctx.beginPath(); ctx.moveTo(X0, Y0); ctx.lineTo(X, Y); ctx.stroke();
      if (Math.hypot(X - X0, Y - Y0) > 4) {
        ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(X - 9 * Math.cos(ang - 0.4), Y - 9 * Math.sin(ang - 0.4)); ctx.lineTo(X - 9 * Math.cos(ang + 0.4), Y - 9 * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill();
      }
      ctx.restore();
    }
    P.clear();
    // original grid (faint) and axes
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, c.w, c.h); ctx.clip();
    var k;
    for (k = -8; k <= 8; k++) {
      seg(k, -8, k, 8, col.line, 1); seg(-8, k, 8, k, col.line, 1);
    }
    seg(-8, 0, 8, 0, col.muted, 1.2); seg(0, -8, 0, 8, col.muted, 1.2);
    // deformed grid
    if (showGrid.get()) {
      for (k = -8; k <= 8; k++) {
        var a0 = T(k, -8), a1 = T(k, 8), b0 = T(-8, k), b1 = T(8, k);
        var cc = k === 0 ? 2 : 1.2;
        seg(a0[0], a0[1], a1[0], a1[1], col.blue, cc);
        seg(b0[0], b0[1], b1[0], b1[1], col.blue, cc);
      }
    }
    // unit square image
    var sq = [[0, 0], [1, 0], [1, 1], [0, 1]].map(function (p) { return T(p[0], p[1]); });
    ctx.save();
    ctx.globalAlpha = 0.25; ctx.fillStyle = col.amber;
    ctx.beginPath(); sq.forEach(function (p, i) { if (i) ctx.lineTo(P.px(p[0]), P.py(p[1])); else ctx.moveTo(P.px(p[0]), P.py(p[1])); }); ctx.closePath(); ctx.fill();
    ctx.globalAlpha = 1; ctx.strokeStyle = col.amber; ctx.lineWidth = 2; ctx.stroke();
    ctx.restore();
    // circle -> ellipse
    var sv = svd(M);
    if (showCircle.get()) {
      var xs = [], ys = [];
      for (k = 0; k <= 120; k++) { var t = k / 120 * 2 * Math.PI, q = T(Math.cos(t), Math.sin(t)); xs.push(q[0]); ys.push(q[1]); }
      var cx = [], cy = [];
      for (k = 0; k <= 120; k++) { var t2 = k / 120 * 2 * Math.PI; cx.push(Math.cos(t2)); cy.push(Math.sin(t2)); }
      P.line(cx, cy, col.muted, 1.5, [4, 4]);
      P.line(xs, ys, col.accent, 2.5);
    }
    if (showSVD.get()) {
      seg(0, 0, sv.s1 * sv.u1[0], sv.s1 * sv.u1[1], col.purple, 3);
      seg(0, 0, sv.s2 * sv.u2[0], sv.s2 * sv.u2[1], col.purple, 3);
      seg(-sv.v1[0] * 1.3, -sv.v1[1] * 1.3, sv.v1[0] * 1.3, sv.v1[1] * 1.3, col.purple, 1, [3, 4]);
      seg(-sv.v2[0] * 1.3, -sv.v2[1] * 1.3, sv.v2[0] * 1.3, sv.v2[1] * 1.3, col.purple, 1, [3, 4]);
    }
    var E = eig(M);
    if (showEig.get()) {
      E.vecs.forEach(function (e) { seg(-e.v[0] * 4, -e.v[1] * 4, e.v[0] * 4, e.v[1] * 4, col.green, 2, [8, 5]); });
    }
    // basis images
    arrow(M[0], M[2], col.red, 3);
    arrow(M[1], M[3], col.green, 3);
    arrow(1, 0, col.muted, 1.5); arrow(0, 1, col.muted, 1.5);
    ctx.restore();
    P.text('M e₁', M[0], M[2], col.red, 'left');
    P.text('M e₂', M[1], M[3], col.green, 'left');
    var det = M[0] * M[3] - M[1] * M[2];
    var h = 'M = [[' + DL.fmt(M[0], 2) + ', ' + DL.fmt(M[1], 2) + '], [' + DL.fmt(M[2], 2) + ', ' + DL.fmt(M[3], 2) + ']] &nbsp; det = <b>' + DL.fmt(det, 3) + '</b> (area scale' + (det < 0 ? ', orientation flipped' : '') + ') &nbsp; trace = ' + DL.fmt(M[0] + M[3], 2) + '<br>';
    h += 'singular values σ₁ = <b>' + DL.fmt(sv.s1, 3) + '</b>, σ₂ = <b>' + DL.fmt(sv.s2, 3) + '</b> &nbsp; condition number σ₁/σ₂ = ' + (sv.s2 > 1e-9 ? DL.fmt(sv.s1 / sv.s2, 2) : '∞ (rank 1)') + '<br>';
    if (E.vecs.length === 0) h += 'eigenvalues: complex, ' + DL.fmt(E.re, 3) + ' ± ' + DL.fmt(E.im, 3) + 'i (no real invariant direction; |λ| = ' + DL.fmt(Math.hypot(E.re, E.im), 3) + ')';
    else h += 'eigenvalues: ' + E.vecs.map(function (e) { return DL.fmt(e.lam, 3); }).join(', ') + (E.vecs.length === 1 ? ' (one real eigen-direction)' : '');
    say(h);
  }
  c.draw = draw;
});
