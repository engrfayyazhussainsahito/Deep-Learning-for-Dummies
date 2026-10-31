DL.register('week18-readiness-radar', function (root) {
  // Self-assessment: one confidence slider per week; radar summary plus a prioritized study plan.
  var names = ['Foundations', 'Backprop', 'Optimization', 'Regularization', 'CNNs', 'Modern vision', 'RNNs', 'Embeddings',
    'Transformers', 'LLMs', 'ViT / CLIP', 'VAE / GAN', 'Diffusion', 'GNNs', 'Deep RL', 'Efficiency', 'Trustworthy DL'];
  var weight = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]; // exam emphasis: Weeks 9 to 17 count double
  var ctrl = DL.controls(root);
  var sl = [];
  var grid = DL.el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:2px 14px;width:100%' });
  ctrl.appendChild(grid);
  names.forEach(function (n, i) {
    var holder = DL.el('div', {});
    grid.appendChild(holder);
    sl.push(DL.slider(holder, { label: 'W' + (i + 1) + ' ' + n, min: 0, max: 5, step: 1, value: 3, format: function (v) { return String(Math.round(v)); } }, draw));
  });
  var row = DL.row(root);
  DL.button(row, 'All 3', function () { sl.forEach(function (s) { s.set(3); }); draw(); });
  DL.button(row, 'Randomize', function () { sl.forEach(function (s) { s.set(1 + Math.floor(Math.random() * 5)); }); draw(); });
  DL.button(row, 'Low on late weeks', function () { sl.forEach(function (s, i) { s.set(i < 8 ? 4 : 2); }); draw(); });
  var c = DL.canvas(root, { aspect: 0.8 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Rate your confidence from 0 (never saw it) to 5 (could teach it) for each week. The radar shows your profile against the target ring at 4; the priority score multiplies the gap to 4 by the exam weight (Weeks 9 to 17 count double). Study the highest-priority weeks first, using the practice exam and the per-week summaries below.' }));
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), ctx = c.ctx, W = c.w, H = c.h, n = names.length;
    ctx.clearRect(0, 0, W, H);
    var cx = W / 2, cy = H / 2 + 4, R = Math.max(40, Math.min(W, H) / 2 - 62), i, k;
    function pt(idx, v) { var a = -Math.PI / 2 + 2 * Math.PI * idx / n; return [cx + Math.cos(a) * R * v / 5, cy + Math.sin(a) * R * v / 5]; }
    ctx.strokeStyle = col.line; ctx.lineWidth = 1;
    for (k = 1; k <= 5; k++) {
      ctx.beginPath();
      for (i = 0; i < n; i++) { var p = pt(i, k); if (i) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]); }
      ctx.closePath(); ctx.stroke();
    }
    for (i = 0; i < n; i++) { var q = pt(i, 5); ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(q[0], q[1]); ctx.stroke(); }
    ctx.setLineDash([5, 4]); ctx.strokeStyle = col.green || col.muted; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (i = 0; i < n; i++) { var t = pt(i, 4); if (i) ctx.lineTo(t[0], t[1]); else ctx.moveTo(t[0], t[1]); }
    ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
    var vals = sl.map(function (s) { return s.get(); });
    ctx.beginPath();
    for (i = 0; i < n; i++) { var u = pt(i, vals[i]); if (i) ctx.lineTo(u[0], u[1]); else ctx.moveTo(u[0], u[1]); }
    ctx.closePath();
    ctx.globalAlpha = 0.22; ctx.fillStyle = col.accent; ctx.fill(); ctx.globalAlpha = 1;
    ctx.strokeStyle = col.accent; ctx.lineWidth = 2.5; ctx.stroke();
    ctx.fillStyle = col.accent;
    for (i = 0; i < n; i++) { var d = pt(i, vals[i]); ctx.beginPath(); ctx.arc(d[0], d[1], 3.5, 0, 7); ctx.fill(); }
    ctx.fillStyle = col.ink || col.muted; ctx.font = '11px sans-serif'; ctx.textBaseline = 'middle';
    for (i = 0; i < n; i++) {
      var a = -Math.PI / 2 + 2 * Math.PI * i / n, lx = cx + Math.cos(a) * (R + 12), ly = cy + Math.sin(a) * (R + 12);
      ctx.textAlign = Math.cos(a) > 0.25 ? 'left' : (Math.cos(a) < -0.25 ? 'right' : 'center');
      ctx.fillText('W' + (i + 1), lx, ly + (Math.abs(Math.cos(a)) <= 0.25 ? (Math.sin(a) > 0 ? 8 : -8) : 0));
    }
    var tot = 0, wsum = 0, pr = [];
    for (i = 0; i < n; i++) {
      tot += vals[i] * weight[i]; wsum += 5 * weight[i];
      pr.push({ i: i, s: Math.max(0, 4 - vals[i]) * weight[i] });
    }
    pr.sort(function (a, b) { return b.s - a.s; });
    var top = pr.filter(function (p) { return p.s > 0; }).slice(0, 4).map(function (p) { return 'W' + (p.i + 1) + ' ' + names[p.i] + ' (' + p.s + ')'; });
    var ready = tot / wsum * 100;
    say('Weighted readiness: <b>' + DL.fmt(ready, 0) + '%</b> &nbsp; ' + (top.length ? 'Study first: <b>' + top.join(', ') + '</b>' : '<b>Every week is at or above the target ring.</b>'));
  }
  c.draw = draw;
});
