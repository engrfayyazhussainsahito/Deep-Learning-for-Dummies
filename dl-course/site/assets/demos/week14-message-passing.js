DL.register('week14-message-passing', function (root) {
  // Editable graph; drag nodes, click two nodes to toggle the edge between them.
  var pos = [[0.12, 0.25], [0.30, 0.12], [0.30, 0.55], [0.48, 0.35], [0.62, 0.12], [0.62, 0.62], [0.80, 0.30], [0.90, 0.62], [0.12, 0.75]];
  var n = pos.length;
  var A = [];
  function zero() { A = []; for (var i = 0; i < n; i++) { A.push([]); for (var j = 0; j < n; j++) A[i].push(0); } }
  zero();
  [[0, 1], [0, 2], [1, 2], [2, 3], [1, 3], [3, 4], [3, 5], [4, 6], [5, 6], [6, 7], [5, 7], [0, 8], [2, 8]].forEach(function (e) { A[e[0]][e[1]] = A[e[1]][e[0]] = 1; });
  var h = [], round = 0, sel = -1;
  var rand = DL.rng(3);

  var ctrl = DL.controls(root);
  var agg = DL.select(ctrl, { label: 'Aggregator', value: 'gcn', options: [
    { value: 'sum', label: 'sum (GIN-like, no self)' }, { value: 'mean', label: 'mean (incl. self)' },
    { value: 'gcn', label: 'GCN: D^-1/2 (A+I) D^-1/2' }, { value: 'max', label: 'max (incl. self)' }] }, draw);
  var init = DL.select(ctrl, { label: 'Initial features', value: 'onehot', options: [
    { value: 'onehot', label: 'one-hot at node 0' }, { value: 'rand', label: 'random' }, { value: 'ones', label: 'all ones' }] }, function () { reset(); });
  var row = DL.row(root);
  DL.button(row, 'Step (one layer)', step);
  DL.button(row, 'Reset', reset);
  var c = DL.canvas(root, { aspect: 0.55 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Drag nodes. Click one node, then another, to add or remove an edge (then press Reset). Press Step and watch information travel exactly one hop per layer. Try "all ones" with sum vs mean vs GCN: only sum reveals the degree; with GCN the values stay near 1 only on regular graphs.' }));

  function reset() {
    h = [];
    for (var i = 0; i < n; i++) h.push(init.get() === 'onehot' ? (i === 0 ? 1 : 0) : init.get() === 'ones' ? 1 : Math.round(rand() * 100) / 100);
    round = 0; draw();
  }
  function deg(i) { var d = 0; for (var j = 0; j < n; j++) d += A[i][j]; return d; }
  function step() {
    var out = [], mode = agg.get(), i, j;
    for (i = 0; i < n; i++) {
      var s = 0, m = -Infinity, k = 0;
      if (mode === 'sum') { for (j = 0; j < n; j++) if (A[i][j]) s += h[j]; out.push(s); continue; }
      if (mode === 'gcn') { var di = deg(i) + 1; s = h[i] / di; for (j = 0; j < n; j++) if (A[i][j]) s += h[j] / Math.sqrt(di * (deg(j) + 1)); out.push(s); continue; }
      s = h[i]; m = h[i]; k = 1;
      for (j = 0; j < n; j++) if (A[i][j]) { s += h[j]; m = Math.max(m, h[j]); k++; }
      out.push(mode === 'mean' ? s / k : m);
    }
    h = out; round++; draw();
  }
  function px(i) { return [pos[i][0] * c.w, pos[i][1] * c.h * 1.25 + c.h * 0.04]; }
  function hit(x, y) { for (var i = 0; i < n; i++) { var p = px(i); if ((p[0] - x) * (p[0] - x) + (p[1] - y) * (p[1] - y) < 22 * 22) return i; } return -1; }
  var grab = -1, moved = false;
  DL.drag(c.canvas, {
    down: function (x, y) { grab = hit(x, y); moved = false; },
    move: function (x, y) { if (grab >= 0) { moved = true; pos[grab] = [Math.min(0.97, Math.max(0.03, x / c.w)), Math.min(0.8, Math.max(0.02, (y - c.h * 0.04) / (c.h * 1.25)))]; draw(); } },
    up: function () {
      if (grab >= 0 && !moved) {
        if (sel < 0) sel = grab;
        else if (sel === grab) sel = -1;
        else { A[sel][grab] = A[grab][sel] = A[sel][grab] ? 0 : 1; sel = -1; }
        draw();
      }
      grab = -1;
    }
  });
  function draw() {
    if (!c.w || !h.length) return;
    var ctx = c.ctx, col = DL.colors(), i, j, mx = 1e-9;
    ctx.clearRect(0, 0, c.w, c.h);
    for (i = 0; i < n; i++) mx = Math.max(mx, Math.abs(h[i]));
    ctx.strokeStyle = col.muted; ctx.lineWidth = 1.5;
    for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) if (A[i][j]) { var a = px(i), b = px(j); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.font = '12px sans-serif';
    var nz = 0, tot = 0;
    for (i = 0; i < n; i++) {
      var p = px(i), t = 0.5 + 0.5 * h[i] / mx;
      if (Math.abs(h[i]) > 1e-9) nz++;
      tot += h[i];
      ctx.beginPath(); ctx.arc(p[0], p[1], 20, 0, 6.2832);
      ctx.fillStyle = DL.cmap.warm(Math.max(0, Math.min(1, Math.abs(h[i]) / mx))); ctx.fill();
      ctx.lineWidth = i === sel ? 3.5 : 1.5; ctx.strokeStyle = i === sel ? col.accent : col.ink; ctx.stroke();
      ctx.fillStyle = col.ink;
      var v = Math.abs(h[i]) >= 1000 ? h[i].toExponential(1) : (Math.round(h[i] * 100) / 100).toString();
      ctx.fillText(v, p[0], p[1]);
      ctx.fillStyle = col.muted; ctx.font = '10px sans-serif'; ctx.fillText('v' + i, p[0], p[1] + 31); ctx.font = '12px sans-serif';
    }
    say('layers applied: <b>' + round + '</b> &nbsp; nodes with non-zero value: <b>' + nz + '/' + n + '</b> &nbsp; total mass: <b>' + DL.fmt(tot, 3) + '</b> &nbsp; max |value|: <b>' + DL.fmt(mx < 1e-8 ? 0 : mx, 3) + '</b>');
  }
  c.draw = draw;
  reset();
});
