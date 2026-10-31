DL.register('week06-map-explorer', function (root) {
  var NGT = 25, NFP = 40, rand = DL.rng(99), cands = [];
  // Fixed random draws, so that the sliders move the data smoothly.
  for (var i = 0; i < NGT; i++) cands.push({ gt: true, found: rand() < 0.92, zs: DL.randn(rand), zi: DL.randn(rand) });
  for (var j = 0; j < NFP; j++) cands.push({ gt: false, found: true, zs: DL.randn(rand), zi: 0 });
  function build(cls, loc) {
    var out = [];
    cands.forEach(function (cd) {
      if (!cd.found) return;
      var s = cd.gt ? 0.5 + 0.38 * cls + 0.16 * cd.zs : 0.5 - 0.34 * cls + 0.16 * cd.zs;
      out.push({ s: DL.clamp(s, 0.01, 0.99), gt: cd.gt, iou: cd.gt ? DL.clamp(0.62 + 0.3 * loc + 0.09 * cd.zi, 0.05, 0.99) : 0 });
    });
    out.sort(function (a, b) { return b.s - a.s; });
    return out;
  }
  // PR curve at a given IoU threshold. Each GT-linked detection is a TP iff IoU >= t (one detection per GT here).
  function pr(dets, t) {
    var tp = 0, fp = 0, rec = [], prec = [];
    dets.forEach(function (d) { if (d.gt && d.iou >= t) tp++; else fp++; rec.push(tp / NGT); prec.push(tp / (tp + fp)); });
    return { rec: rec, prec: prec, tp: tp, fp: fp };
  }
  function apAll(rec, prec) { // all-point interpolation: area under the monotone envelope of precision
    var r = [0].concat(rec, [1]), p = [0].concat(prec, [0]);
    for (var i = p.length - 2; i >= 0; i--) p[i] = Math.max(p[i], p[i + 1]);
    var ap = 0; for (var k = 1; k < r.length; k++) ap += (r[k] - r[k - 1]) * p[k];
    return { ap: ap, env: { r: r, p: p } };
  }
  var ctrl = DL.controls(root);
  var cls = DL.slider(ctrl, { label: 'Classifier quality (score separation)', min: 0, max: 1, step: 0.01, value: 0.7, format: function (v) { return (+v).toFixed(2); } }, draw);
  var loc = DL.slider(ctrl, { label: 'Localization quality (typical IoU)', min: 0, max: 1, step: 0.01, value: 0.6, format: function (v) { return (+v).toFixed(2); } }, draw);
  var it = DL.slider(ctrl, { label: 'IoU threshold for a true positive', min: 0.3, max: 0.95, step: 0.05, value: 0.5, format: function (v) { return (+v).toFixed(2); } }, draw);
  var st = DL.slider(ctrl, { label: 'Score threshold (operating point)', min: 0, max: 1, step: 0.01, value: 0.5, format: function (v) { return (+v).toFixed(2); } }, draw);
  var c = DL.canvas(root, { aspect: 0.55 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'One class, 25 ground-truth objects, 40 spurious detections. Detections are sorted by score; walking down the list traces the precision-recall curve (jagged line). AP is the area under its monotone envelope (shaded). Try: raise the IoU threshold and watch AP fall when localization is mediocre even though the classifier is unchanged; then compare with the COCO-style average over IoU 0.50 to 0.95.' }));
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), dets = build(cls.get(), loc.get()), t = it.get();
    var q = pr(dets, t), a = apAll(q.rec, q.prec);
    var P = DL.plot(c, { xmin: 0, xmax: 1, ymin: 0, ymax: 1.02, xlabel: 'recall', ylabel: 'precision', pad: { l: 46 } });
    P.clear(); P.grid(); P.axes();
    // shaded area
    var ctx = c.ctx; ctx.save(); ctx.globalAlpha = 0.18; ctx.fillStyle = col.accent; ctx.beginPath(); ctx.moveTo(P.px(0), P.py(0));
    for (var k = 0; k < a.env.r.length; k++) { ctx.lineTo(P.px(a.env.r[k]), P.py(a.env.p[k])); if (k + 1 < a.env.r.length) ctx.lineTo(P.px(a.env.r[k + 1]), P.py(a.env.p[k])); }
    ctx.lineTo(P.px(1), P.py(0)); ctx.closePath(); ctx.fill(); ctx.restore();
    P.line([0].concat(q.rec), [q.prec[0] || 1].concat(q.prec), col.blue, 1.6);
    var ex = [], ey = []; for (var m = 0; m < a.env.r.length; m++) { ex.push(a.env.r[m]); ey.push(a.env.p[m]); }
    P.line(ex, ey, col.accent, 2.5);
    // operating point
    var n = 0; while (n < dets.length && dets[n].s >= st.get()) n++;
    var opR = n > 0 ? q.rec[n - 1] : 0, opP = n > 0 ? q.prec[n - 1] : 1;
    P.points([opR], [opP], col.red, 6);
    P.text('operating point', Math.min(0.75, opR + 0.02), Math.min(0.97, opP + 0.04), col.red, 'left');
    // COCO-style average over IoU thresholds
    var tot = 0, ths = []; for (var u = 0.5; u < 0.951; u += 0.05) { ths.push(u); tot += apAll.apply(null, (function () { var qq = pr(dets, u); return [qq.rec, qq.prec]; })()).ap; }
    var nTP = n > 0 ? Math.round(opR * NGT) : 0, nDet = n;
    var f1 = opP + opR > 0 ? 2 * opP * opR / (opP + opR) : 0;
    say('AP@IoU ' + DL.fmt(t, 2) + ' = <b>' + DL.fmt(a.ap, 3) + '</b> &nbsp; | &nbsp; COCO-style AP@[.5:.95] = <b>' + DL.fmt(tot / ths.length, 3) + '</b> &nbsp; | &nbsp; AP@.50 = <b>' + DL.fmt(apAll.apply(null, (function () { var qq = pr(dets, 0.5); return [qq.rec, qq.prec]; })()).ap, 3) + '</b><br>' +
      'At score &ge; ' + DL.fmt(st.get(), 2) + ': <b>' + nDet + '</b> detections, <b>' + nTP + '</b> true positives, precision <b>' + DL.fmt(opP, 2) + '</b>, recall <b>' + DL.fmt(opR, 2) + '</b>, F1 <b>' + DL.fmt(f1, 2) + '</b>. ');
  }
  c.draw = draw;
});
