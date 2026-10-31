DL.register('week11-vit-patchify', function (root) {
  var SIZE = 224, sel = { r: 3, c: 4 };
  var ctrl = DL.controls(root);
  var psSel = DL.select(ctrl, { label: 'Patch size P', value: '32', options: [
    { value: '56', label: '56 (4 x 4 grid)' }, { value: '32', label: '32 (7 x 7 grid)' }, { value: '16', label: '16 (14 x 14 grid)' }, { value: '14', label: '14 (16 x 16 grid)' }, { value: '8', label: '8 (28 x 28 grid)' }] }, draw);
  var modeSel = DL.select(ctrl, { label: 'Position embedding', value: '2d', options: [
    { value: '2d', label: '2-D sin-cos (row half + column half)' }, { value: '1d', label: '1-D sin-cos on raster index' }, { value: 'none', label: 'none (permutation-blind)' }] }, draw);
  var dimS = DL.slider(ctrl, { label: 'Embedding dim D', min: 8, max: 256, step: 8, value: 64 }, draw);
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Click or drag on the image to choose a patch. The right panel shows the cosine similarity between that patch\'s position embedding and every other patch. With 2-D embeddings the bright region is a blob around the chosen patch; with 1-D raster embeddings it forms stripes (neighbouring rows are far apart in the sequence); with no embedding every patch looks identical to attention.' }));
  var geo = { x1: 0, y1: 0, s: 1, x2: 0 };
  function grid() { return Math.round(SIZE / parseInt(psSel.get(), 10)); }
  function pe1(i, D) { var v = [], k; for (k = 0; k < D / 2; k++) { var w = 1 / Math.pow(10000, 2 * k / D); v.push(Math.sin(i * w), Math.cos(i * w)); } return v; }
  function pe(r, cc, G, D) {
    var m = modeSel.get(), v;
    if (m === 'none') { v = []; for (var k = 0; k < D; k++) v.push(1); return v; }
    if (m === '1d') return pe1(r * G + cc, D);
    return pe1(r, D / 2).concat(pe1(cc, D / 2));
  }
  function cos(a, b) { var s = 0, na = 0, nb = 0; for (var i = 0; i < a.length; i++) { s += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return s / (Math.sqrt(na * nb) + 1e-12); }
  function paintImage(x, y, s) {
    var ctx = c.ctx;
    ctx.save();
    ctx.beginPath(); ctx.rect(x, y, s, s); ctx.clip();
    var g = ctx.createLinearGradient(x, y, x, y + s); g.addColorStop(0, '#7fb3e6'); g.addColorStop(0.62, '#cfe6f7'); g.addColorStop(0.62, '#7aa858'); g.addColorStop(1, '#4f7a3a');
    ctx.fillStyle = g; ctx.fillRect(x, y, s, s);
    ctx.fillStyle = '#f2c94c'; ctx.beginPath(); ctx.arc(x + s * 0.8, y + s * 0.2, s * 0.09, 0, 6.2832); ctx.fill();
    ctx.fillStyle = '#b5513a'; ctx.fillRect(x + s * 0.18, y + s * 0.4, s * 0.3, s * 0.26);
    ctx.fillStyle = '#6b2f23'; ctx.beginPath(); ctx.moveTo(x + s * 0.14, y + s * 0.4); ctx.lineTo(x + s * 0.33, y + s * 0.22); ctx.lineTo(x + s * 0.52, y + s * 0.4); ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#f5f0e0'; ctx.fillRect(x + s * 0.28, y + s * 0.5, s * 0.07, s * 0.16);
    ctx.restore();
  }
  function draw() {
    if (!c.w) return;
    var ctx = c.ctx, col = DL.colors(), G = grid(), D = Math.round(dimS.get() / 4) * 4;
    ctx.clearRect(0, 0, c.w, c.h);
    sel.r = Math.min(sel.r, G - 1); sel.c = Math.min(sel.c, G - 1);
    var s = Math.max(60, Math.min(c.h - 30, (c.w - 30) / 2));
    geo.x1 = 10; geo.y1 = 22; geo.s = s; geo.x2 = c.w - s - 10;
    paintImage(geo.x1, geo.y1, s);
    var cell = s / G, i, j;
    ctx.save(); ctx.strokeStyle = 'rgba(255,255,255,0.75)'; ctx.lineWidth = 1;
    ctx.beginPath();
    for (i = 0; i <= G; i++) { ctx.moveTo(geo.x1 + i * cell, geo.y1); ctx.lineTo(geo.x1 + i * cell, geo.y1 + s); ctx.moveTo(geo.x1, geo.y1 + i * cell); ctx.lineTo(geo.x1 + s, geo.y1 + i * cell); }
    ctx.stroke();
    ctx.strokeStyle = col.accent; ctx.lineWidth = 3; ctx.strokeRect(geo.x1 + sel.c * cell, geo.y1 + sel.r * cell, cell, cell);
    ctx.restore();
    var ref = pe(sel.r, sel.c, G, D), M = [];
    for (i = 0; i < G; i++) { var row = []; for (j = 0; j < G; j++) row.push(cos(ref, pe(i, j, G, D))); M.push(row); }
    DL.heatmap(ctx, M, geo.x2, geo.y1, s, s, { min: -1, max: 1, cmap: DL.cmap.diverge, border: true });
    ctx.save(); ctx.strokeStyle = col.ink; ctx.lineWidth = 2; ctx.strokeRect(geo.x2 + sel.c * cell, geo.y1 + sel.r * cell, cell, cell); ctx.restore();
    ctx.fillStyle = col.ink; ctx.font = '12px ' + getComputedStyle(document.body).getPropertyValue('--font-sans'); ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('image 224 x 224 split into patches', geo.x1, 10); ctx.fillText('cosine(pos[selected], pos[j])', geo.x2, 10);
    var N = G * G, pd = parseInt(psSel.get(), 10);
    say('P = <b>' + pd + '</b> &nbsp; N = (224/P)² = <b>' + N + '</b> patches &nbsp; sequence = N + 1 = <b>' + (N + 1) + '</b> tokens<br>' +
      'raw patch vector: P·P·3 = <b>' + (pd * pd * 3) + '</b> numbers → linear map to D &nbsp; attention matrix per head: <b>' + (N + 1) + ' × ' + (N + 1) + ' = ' + ((N + 1) * (N + 1)) + '</b> entries<br>' +
      'selected patch: row ' + sel.r + ', col ' + sel.c + ', raster index ' + (sel.r * G + sel.c));
  }
  function pick(x, y) {
    var G = grid(), cell = geo.s / G;
    var c0 = Math.floor((x - geo.x1) / cell), r0 = Math.floor((y - geo.y1) / cell);
    if (c0 >= 0 && c0 < G && r0 >= 0 && r0 < G) { sel.r = r0; sel.c = c0; draw(); }
  }
  DL.drag(c.canvas, { down: pick, move: pick });
  c.draw = draw;
});
