DL.register('week09-mask-cost', function (root) {
  var ctrl = DL.controls(root);
  var pat = DL.select(ctrl, { label: 'Pattern', value: 'causal', options: [
    { value: 'full', label: 'Full (encoder)' }, { value: 'causal', label: 'Causal (decoder)' },
    { value: 'window', label: 'Sliding window' }, { value: 'dilated', label: 'Dilated window' },
    { value: 'global', label: 'Window + global tokens' }, { value: 'prefix', label: 'Prefix-LM' }] }, draw);
  var n = DL.slider(ctrl, { label: 'Sequence length n', min: 8, max: 32, step: 1, value: 16 }, draw);
  var w = DL.slider(ctrl, { label: 'Window / prefix w', min: 1, max: 16, step: 1, value: 4 }, draw);
  var g = DL.slider(ctrl, { label: 'Global tokens g', min: 0, max: 4, step: 1, value: 1 }, draw);
  var causalToggle = DL.toggle(ctrl, 'Also apply causal mask (window/dilated/global)', true, draw);
  var c = DL.canvas(root, { aspect: 0.62 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Rows are queries, columns keys; colored cells are score entries that are computed. The readout counts allowed entries. Right panel: share of per-layer FLOPs spent in attention as n grows (d = 1024), when only the allowed entries are computed (4·d FLOPs each for QK^T and AV), versus the 24·n·d² of the dense projections and FFN, which is linear in n. Try the sliding window with w = 4: the curve stays low where full attention dominates.' }));
  function allowed(i, j, m, W, G, cz) {
    var inc;
    if (m === 'full') return true;
    if (m === 'causal') return j <= i;
    if (m === 'prefix') return j <= i || j < W;
    if (m === 'window') inc = Math.abs(i - j) < W;
    else if (m === 'dilated') inc = Math.abs(i - j) % 2 === 0 && Math.abs(i - j) < 2 * W;
    else inc = Math.abs(i - j) < W || i < G || j < G;
    return inc && (!cz || j <= i || (m === 'global' && i < G));
  }
  function draw() {
    var ctx = c.ctx, col = DL.colors(), N = Math.round(n.get()), W = Math.round(w.get()), G = Math.round(g.get()), m = pat.get(), cz = causalToggle.get();
    ctx.clearRect(0, 0, c.w, c.h);
    var side = Math.min(c.h - 20, c.w * 0.45), x0 = 12, y0 = 10, cell = side / N, i, j, cnt = 0;
    for (i = 0; i < N; i++) for (j = 0; j < N; j++) {
      var a = allowed(i, j, m, W, G, cz); if (a) cnt++;
      ctx.fillStyle = a ? col.accent : col.soft; ctx.globalAlpha = a ? 0.85 : 1;
      ctx.fillRect(x0 + j * cell, y0 + i * cell, Math.max(1, cell - 1), Math.max(1, cell - 1));
    }
    ctx.globalAlpha = 1;
    var P = DL.plot(c, { xmin: 7, xmax: 15, ymin: 0, ymax: 1, pad: { l: x0 + side + 60, r: 14, t: 12, b: 36 }, xlabel: 'log2 n  (n = 128 ... 32768)', ylabel: 'attention share of FLOPs' });
    var d = 1024, xs = [], ys = [], ys2 = [], e;
    var wEff = cz ? W : 2 * W - 1;
    for (e = 7; e <= 15; e += 0.25) {
      var nn = Math.pow(2, e), full = 4 * nn * nn * d, lin = 24 * nn * d * d, dens;
      if (m === 'full') dens = 1; else if (m === 'causal') dens = 0.5; else if (m === 'prefix') dens = 0.5 + 0.5 * Math.min(1, W / nn);
      else if (m === 'window' || m === 'dilated') dens = Math.min(1, wEff / nn);
      else dens = Math.min(1, (wEff + G * (cz ? 1 : 2)) / nn);
      var att = full * dens; xs.push(e); ys.push(att / (att + lin)); ys2.push(full / (full + lin));
    }
    P.grid(); P.axes();
    P.line(xs, ys2, col.muted, 1.8, [5, 4]); P.line(xs, ys, col.accent, 2.6);
    P.text('dashed: full attention', 7.2, 0.93, col.muted, 'left'); P.text('solid: selected pattern', 7.2, 0.86, col.accent, 'left');
    say('Allowed entries: <b>' + cnt + '</b> of ' + (N * N) + ' (' + DL.fmt(100 * cnt / (N * N), 1) + '% density). With full attention the attention share reaches 50% at n = 6d = 6144 for d = 1024.');
  }
  c.draw = draw;
  draw();
});
