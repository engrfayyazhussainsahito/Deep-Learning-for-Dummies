DL.register('week04-mixup', function (root) {
  var ctrl = DL.controls(root);
  var mode = DL.select(ctrl, { label: 'Method', value: 'mixup', options: [{ value: 'mixup', label: 'mixup' }, { value: 'cutmix', label: 'CutMix' }] }, draw);
  var al = DL.slider(ctrl, { label: 'Beta α', min: 0.1, max: 4, step: 0.1, value: 1 }, resample);
  var lamS = DL.slider(ctrl, { label: 'λ (override)', min: 0, max: 1, step: 0.01, value: 0.5 }, function () { useOverride = true; draw(); });
  var row = DL.row(root);
  DL.button(row, 'Draw λ ~ Beta(α, α) and new box', resample);
  var useOverride = false, lam = 0.5, box = [0.3, 0.2];
  var c = DL.canvas(root, { aspect: 0.4 });
  var say = DL.readout(root);
  var rnd = DL.rng(5);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Two synthetic 32×32 images (A: circle, class 0; B: diagonal stripes, class 1). mixup blends every pixel with weight λ; CutMix pastes a box from B whose area is 1-λ, and the label weights follow the same fractions. Small α concentrates λ near 0 or 1 (mild mixing), α ≥ 1 mixes more strongly.' }));
  var S = 32;
  function imgA(x, y) { var d = Math.hypot(x - 16, y - 16); return d < 9 ? 1 : 0.1; }
  function imgB(x, y) { return (Math.floor((x + y) / 4) % 2) ? 0.9 : 0.15; }
  function gamma(k) { // Marsaglia-Tsang, k>0
    if (k < 1) return gamma(k + 1) * Math.pow(rnd(), 1 / k);
    var d = k - 1 / 3, cc = 1 / Math.sqrt(9 * d);
    for (;;) { var x = DL.randn(rnd), v = 1 + cc * x; if (v <= 0) continue; v = v * v * v; var u = rnd(); if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v; }
  }
  function resample() { var a = al.get(), g1 = gamma(a), g2 = gamma(a); lam = g1 / (g1 + g2); lamS.set(lam); useOverride = false; box = [rnd(), rnd()]; draw(); }
  function drawImg(ctx, f, x0, y0, sz, col) {
    var px = sz / S;
    for (var y = 0; y < S; y++) for (var x = 0; x < S; x++) { var v = DL.clamp(f(x, y), 0, 1), g = Math.round(40 + 200 * v); ctx.fillStyle = 'rgb(' + g + ',' + g + ',' + g + ')'; ctx.fillRect(x0 + x * px, y0 + y * px, px + 0.6, px + 0.6); }
    ctx.strokeStyle = col.line; ctx.strokeRect(x0, y0, sz, sz);
  }
  function draw() {
    var col = DL.colors(), ctx = c.ctx, m = mode.get(), l = lamS.get();
    ctx.clearRect(0, 0, c.w, c.h);
    var sz = Math.min((c.w - 60) / 3, c.h - 50), gap = (c.w - 3 * sz) / 4, y0 = 24, xs = [gap, 2 * gap + sz, 3 * gap + 2 * sz];
    var side = Math.round(S * Math.sqrt(1 - l)), bx = Math.round(box[0] * (S - side)), by = Math.round(box[1] * (S - side));
    var inBox = function (x, y) { return x >= bx && x < bx + side && y >= by && y < by + side; };
    var lamEff = m === 'cutmix' ? 1 - (side * side) / (S * S) : l;
    drawImg(ctx, imgA, xs[0], y0, sz, col); drawImg(ctx, imgB, xs[1], y0, sz, col);
    drawImg(ctx, m === 'mixup' ? function (x, y) { return l * imgA(x, y) + (1 - l) * imgB(x, y); } : function (x, y) { return inBox(x, y) ? imgB(x, y) : imgA(x, y); }, xs[2], y0, sz, col);
    if (m === 'cutmix') { var px = sz / S; ctx.strokeStyle = col.red; ctx.lineWidth = 2; ctx.strokeRect(xs[2] + bx * px, y0 + by * px, side * px, side * px); }
    ctx.fillStyle = col.ink; ctx.font = '13px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('A (class 0)', xs[0] + sz / 2, 14); ctx.fillText('B (class 1)', xs[1] + sz / 2, 14); ctx.fillText(m === 'mixup' ? 'mixed image' : 'CutMix image', xs[2] + sz / 2, 14);
    var by2 = y0 + sz + 10, bw = sz;
    ctx.fillStyle = col.blue; ctx.fillRect(xs[2], by2, bw * lamEff, 14); ctx.fillStyle = col.amber; ctx.fillRect(xs[2] + bw * lamEff, by2, bw * (1 - lamEff), 14);
    ctx.fillStyle = col.muted; ctx.fillText('soft label', xs[2] + sz / 2, by2 + 30);
    say('mixed label = <b>' + DL.fmt(lamEff, 2) + '</b>·A + <b>' + DL.fmt(1 - lamEff, 2) + '</b>·B &nbsp; (loss = ' + DL.fmt(lamEff, 2) + '·CE(ŷ, A) + ' + DL.fmt(1 - lamEff, 2) + '·CE(ŷ, B))' + (m === 'cutmix' ? ' &nbsp; box ' + side + '×' + side + ' px' : ''));
  }
  c.draw = draw;
});
