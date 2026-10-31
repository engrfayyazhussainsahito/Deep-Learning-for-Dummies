DL.register('week04-dropout', function (root) {
  var ctrl = DL.controls(root);
  var pd = DL.slider(ctrl, { label: 'Drop probability p', min: 0, max: 0.9, step: 0.05, value: 0.5 }, draw);
  var nu = DL.slider(ctrl, { label: 'Hidden units', min: 4, max: 200, step: 1, value: 50 }, draw);
  var row = DL.row(root);
  var inv = DL.toggle(row, 'Inverted scaling (divide kept units by 1-p)', true, draw);
  DL.button(row, 'New masks', function () { seed++; draw(); });
  var c = DL.canvas(root, { aspect: 0.62 });
  var say = DL.readout(root);
  var seed = 1;
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'A fixed hidden vector h (positive units) feeds a linear read-out y = wᵀ(m ⊙ h). Top: one random mask (grey = dropped). Bottom: histogram of y over 3000 masks versus the deterministic eval-time output (green line). With inverted scaling the mean matches; switch it off and the train-time mean shrinks by 1-p. Note how variance grows with p and shrinks with more units.' }));
  function draw() {
    var col = DL.colors(), p = pd.get(), N = Math.round(nu.get()), useInv = inv.get(), rnd = DL.rng(77), i;
    var h = [], w = [];
    for (i = 0; i < N; i++) { h.push(0.5 + rnd()); w.push(DL.randn(rnd) * 0.5 + 0.3); }
    var ydet = 0; for (i = 0; i < N; i++) ydet += w[i] * h[i];
    var r2 = DL.rng(seed * 31 + 5), scale = useInv ? 1 / (1 - p) : 1;
    var mask = [], ym = 0;
    for (i = 0; i < N; i++) { var keep = r2() >= p; mask.push(keep); if (keep) ym += w[i] * h[i] * scale; }
    var ys = [], M = 3000, r3 = DL.rng(seed * 101 + 9), s1 = 0, s2 = 0;
    for (var k = 0; k < M; k++) { var y = 0; for (i = 0; i < N; i++) if (r3() >= p) y += w[i] * h[i] * scale; ys.push(y); s1 += y; s2 += y * y; }
    var mean = s1 / M, sd = Math.sqrt(Math.max(0, s2 / M - mean * mean));
    var ctx = c.ctx; ctx.clearRect(0, 0, c.w, c.h);
    // units strip
    var cols = Math.min(N, Math.floor((c.w - 40) / 14)), rows = Math.ceil(N / cols), sz = Math.min(12, (c.w - 40) / cols - 2);
    for (i = 0; i < N; i++) { var cx = 20 + (i % cols) * (sz + 2), cy = 14 + Math.floor(i / cols) * (sz + 2); ctx.fillStyle = mask[i] ? col.blue : col.line; ctx.fillRect(cx, cy, sz, sz); }
    var top = 14 + rows * (sz + 2) + 24, H = c.h - top - 34, L = 44, R = c.w - 14;
    var lo = Math.min(ydet, mean) - 4 * Math.max(sd, 0.05), hi = Math.max(ydet, mean) + 4 * Math.max(sd, 0.05), B = 40, bins = new Array(B).fill(0);
    ys.forEach(function (y) { var b = Math.floor((y - lo) / (hi - lo) * B); if (b >= 0 && b < B) bins[b]++; });
    var mx = Math.max.apply(null, bins) || 1;
    for (i = 0; i < B; i++) { var bh = bins[i] / mx * H; ctx.fillStyle = col.accent + 'aa'; ctx.fillRect(L + i * (R - L) / B, top + H - bh, (R - L) / B - 1, bh); }
    var X = function (v) { return L + (v - lo) / (hi - lo) * (R - L); };
    ctx.strokeStyle = col.green; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(X(ydet), top); ctx.lineTo(X(ydet), top + H); ctx.stroke();
    ctx.strokeStyle = col.red; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(X(mean), top); ctx.lineTo(X(mean), top + H); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = col.muted; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('output y over random masks (green = eval output, red dashed = train mean)', (L + R) / 2, top + H + 22);
    ctx.textAlign = 'left'; ctx.fillText('mask sample: ' + mask.filter(Boolean).length + ' of ' + N + ' kept, y = ' + DL.fmt(ym, 2), 20, 12 + rows * (sz + 2) + 14);
    say('eval output <b>' + DL.fmt(ydet, 2) + '</b> &nbsp; train mean <b>' + DL.fmt(mean, 2) + '</b> &nbsp; train std <b>' + DL.fmt(sd, 2) + '</b> &nbsp; relative noise ' + DL.fmt(sd / Math.abs(ydet), 3) + (useInv ? '' : ' &nbsp; <b>train/eval mismatch: expected ratio ' + DL.fmt(1 - p, 2) + '</b>'));
  }
  c.draw = draw;
});
