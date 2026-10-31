DL.register('week06-iou-nms', function (root) {
  // ---------- geometry (boxes are {x1,y1,x2,y2,s}) in a 160 x 100 scene
  var W = 160, H = 100;
  function area(b) { return Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1); }
  function iou(a, b) {
    var iw = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1), ih = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
    var inter = iw > 0 && ih > 0 ? iw * ih : 0, uni = area(a) + area(b) - inter;
    return uni > 0 ? inter / uni : 0;
  }
  function nmsHard(boxes, thr, minScore) {
    var idx = boxes.map(function (_, i) { return i; }).filter(function (i) { return boxes[i].s >= minScore; });
    idx.sort(function (i, j) { return boxes[j].s - boxes[i].s; });
    var keep = [], state = boxes.map(function () { return 'low'; });
    idx.forEach(function (i) { state[i] = 'pending'; });
    idx.forEach(function (i) {
      if (state[i] !== 'pending') return;
      state[i] = 'keep'; keep.push(i);
      idx.forEach(function (j) { if (state[j] === 'pending' && iou(boxes[i], boxes[j]) > thr) state[j] = 'drop'; });
    });
    return { state: state, score: boxes.map(function (b) { return b.s; }) };
  }
  function nmsSoft(boxes, sigma, minScore) {
    var sc = boxes.map(function (b) { return b.s; }), done = boxes.map(function () { return false; });
    var state = boxes.map(function () { return 'drop'; });
    for (var it = 0; it < boxes.length; it++) {
      var m = -1;
      for (var i = 0; i < boxes.length; i++) if (!done[i] && (m < 0 || sc[i] > sc[m])) m = i;
      if (m < 0 || sc[m] < minScore) break;
      done[m] = true; state[m] = 'keep';
      for (var j = 0; j < boxes.length; j++) if (!done[j]) { var o = iou(boxes[m], boxes[j]); sc[j] *= Math.exp(-o * o / Math.max(1e-6, sigma)); }
    }
    return { state: state, score: sc };
  }
  function scene(seed) {
    var r = DL.rng(seed), out = [], centers = [[38, 50, 40, 52], [92, 44, 34, 40], [128, 66, 26, 30]];
    centers.forEach(function (c) {
      var n = 5 + Math.floor(r() * 3);
      for (var k = 0; k < n; k++) {
        var cx = c[0] + (r() - 0.5) * 8, cy = c[1] + (r() - 0.5) * 8, w = c[2] * (0.85 + 0.3 * r()), h = c[3] * (0.85 + 0.3 * r());
        out.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, s: 0.35 + 0.6 * r() });
      }
    });
    for (var f = 0; f < 3; f++) { var cx2 = 10 + 140 * r(), cy2 = 10 + 80 * r(); out.push({ x1: cx2 - 8, y1: cy2 - 8, x2: cx2 + 8, y2: cy2 + 8, s: 0.05 + 0.3 * r() }); }
    return out;
  }

  // ---------- UI
  var ctrl = DL.controls(root);
  var mode = DL.select(ctrl, { label: 'Mode', value: 'iou', options: [{ value: 'iou', label: 'IoU: drag the box' }, { value: 'nms', label: 'Non-maximum suppression' }] }, function () { sync(); draw(); });
  var bw = DL.slider(ctrl, { label: 'Box B width', min: 10, max: 90, step: 1, value: 50 }, function () { draw(); });
  var bh = DL.slider(ctrl, { label: 'Box B height', min: 10, max: 80, step: 1, value: 40 }, function () { draw(); });
  var thr = DL.slider(ctrl, { label: 'IoU threshold', min: 0.1, max: 0.9, step: 0.01, value: 0.5 }, function () { draw(); });
  var smin = DL.slider(ctrl, { label: 'Score threshold', min: 0, max: 0.9, step: 0.01, value: 0.1 }, function () { draw(); });
  var meth = DL.select(ctrl, { label: 'Suppression', value: 'hard', options: [{ value: 'hard', label: 'Hard NMS' }, { value: 'soft', label: 'Soft-NMS (Gaussian)' }] }, function () { draw(); });
  var row = DL.row(root);
  var seed = 7, boxes = scene(seed);
  DL.button(row, 'New scene', function () { seed += 1; boxes = scene(seed); draw(); });
  var c = DL.canvas(root, { aspect: 0.55 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'IoU mode: drag the orange box over the blue ground-truth box and watch intersection and union. NMS mode: three objects each produce several overlapping detections plus a few weak false positives; lower the IoU threshold to suppress more, raise it and duplicates survive. Soft-NMS decays scores by exp(-IoU^2/sigma) (sigma = 0.5 here) instead of deleting boxes.' }));
  var B = { cx: 95, cy: 55 }, A = { x1: 40, y1: 25, x2: 100, y2: 75 };
  function sync() { var m = mode.get() === 'iou'; bw.el.style.display = m ? '' : 'none'; bh.el.style.display = m ? '' : 'none'; thr.el.style.display = m ? 'none' : ''; smin.el.style.display = m ? 'none' : ''; meth.el.style.display = m ? 'none' : ''; }
  sync();
  function sx(x) { return x / W * c.w; } function sy(y) { return y / H * c.h; }
  function rect(b, stroke, fill, dash, lw) {
    var ctx = c.ctx; ctx.save();
    if (fill) { ctx.globalAlpha = 0.25; ctx.fillStyle = fill; ctx.fillRect(sx(b.x1), sy(b.y1), sx(b.x2) - sx(b.x1), sy(b.y2) - sy(b.y1)); ctx.globalAlpha = 1; }
    ctx.strokeStyle = stroke; ctx.lineWidth = lw || 2; if (dash) ctx.setLineDash(dash);
    ctx.strokeRect(sx(b.x1), sy(b.y1), sx(b.x2) - sx(b.x1), sy(b.y2) - sy(b.y1)); ctx.restore();
  }
  function label(t, x, y, col) { var ctx = c.ctx; ctx.save(); ctx.fillStyle = col; ctx.font = '11px sans-serif'; ctx.fillText(t, x, y); ctx.restore(); }
  function drawIoU(col) {
    var w = bw.get(), h = bh.get();
    B.cx = DL.clamp(B.cx, 0, W); B.cy = DL.clamp(B.cy, 0, H);
    var Bb = { x1: B.cx - w / 2, y1: B.cy - h / 2, x2: B.cx + w / 2, y2: B.cy + h / 2 };
    var ix1 = Math.max(A.x1, Bb.x1), iy1 = Math.max(A.y1, Bb.y1), ix2 = Math.min(A.x2, Bb.x2), iy2 = Math.min(A.y2, Bb.y2);
    var hasI = ix2 > ix1 && iy2 > iy1;
    rect(A, col.blue, col.blue); rect(Bb, col.amber, col.amber);
    if (hasI) rect({ x1: ix1, y1: iy1, x2: ix2, y2: iy2 }, col.green, col.green, null, 2.5);
    label('A (ground truth)', sx(A.x1) + 4, sy(A.y1) - 5, col.blue); label('B (prediction)', sx(Bb.x1) + 4, sy(Bb.y2) + 13, col.amber);
    var I = hasI ? (ix2 - ix1) * (iy2 - iy1) : 0, U = area(A) + area(Bb) - I, v = U > 0 ? I / U : 0;
    say('|A| = <b>' + DL.fmt(area(A), 0) + '</b>, |B| = <b>' + DL.fmt(area(Bb), 0) + '</b>, intersection = <b>' + DL.fmt(I, 0) + '</b>, union = <b>' + DL.fmt(U, 0) + '</b><br>IoU = <b>' + DL.fmt(v, 3) + '</b> &nbsp; (' + (v >= 0.75 ? 'tight match' : v >= 0.5 ? 'counts as a hit at IoU 0.5 but not at 0.75' : 'a miss at the usual 0.5 threshold') + '). Note a box half the size of A, fully inside it, has IoU = 0.5.');
  }
  function drawNMS(col) {
    var r = meth.get() === 'hard' ? nmsHard(boxes, thr.get(), smin.get()) : nmsSoft(boxes, 0.5, smin.get());
    var kept = 0, dropped = 0;
    boxes.forEach(function (b, i) {
      var st = r.state[i];
      if (st === 'keep') { rect(b, col.green, col.green, null, 2.5); kept++; label(DL.fmt(r.score[i], 2), sx(b.x1) + 3, sy(b.y1) + 12, col.green); }
      else if (st === 'drop') { rect(b, col.red, null, [4, 3], 1.2); dropped++; }
      else rect(b, col.muted, null, [2, 4], 1);
    });
    var low = boxes.length - kept - dropped;
    say('<b>' + boxes.length + '</b> raw detections: <b>' + kept + '</b> kept (solid green), <b>' + dropped + '</b> suppressed (dashed red), <b>' + low + '</b> below score threshold (dotted grey). Ground truth has 3 objects' + (kept === 3 ? ': exactly right.' : kept > 3 ? ': duplicates or false positives remain.' : ': an object has been lost.'));
  }
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), ctx = c.ctx;
    ctx.clearRect(0, 0, c.w, c.h);
    ctx.strokeStyle = col.line; ctx.lineWidth = 1; ctx.strokeRect(0.5, 0.5, c.w - 1, c.h - 1);
    if (mode.get() === 'iou') drawIoU(col); else drawNMS(col);
  }
  DL.drag(c.canvas, {
    down: function (x, y) { if (mode.get() === 'iou') { B.cx = x / c.w * W; B.cy = y / c.h * H; draw(); } },
    move: function (x, y) { if (mode.get() === 'iou') { B.cx = x / c.w * W; B.cy = y / c.h * H; draw(); } }
  });
  c.draw = draw;
});
