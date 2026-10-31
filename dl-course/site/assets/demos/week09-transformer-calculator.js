DL.register('week09-transformer-calculator', function (root) {
  var PRESETS = {
    gpt2s: { L: 12, d: 768, V: 50257, ctx: 1024, ff: 4, tie: true, learned: true },
    gpt2m: { L: 24, d: 1024, V: 50257, ctx: 1024, ff: 4, tie: true, learned: true },
    gpt2xl: { L: 48, d: 1600, V: 50257, ctx: 1024, ff: 4, tie: true, learned: true },
    gpt3: { L: 96, d: 12288, V: 50257, ctx: 2048, ff: 4, tie: true, learned: true },
    base: { L: 6, d: 512, V: 37000, ctx: 512, ff: 4, tie: true, learned: false }
  };
  var ctrl = DL.controls(root);
  var pre = DL.select(ctrl, { label: 'Preset', value: 'gpt2s', options: [
    { value: 'custom', label: 'Custom' }, { value: 'gpt2s', label: 'GPT-2 small' }, { value: 'gpt2m', label: 'GPT-2 medium' },
    { value: 'gpt2xl', label: 'GPT-2 XL' }, { value: 'gpt3', label: 'GPT-3 175B shape' }, { value: 'base', label: 'Decoder, Transformer-base widths' }] }, function (v) { apply(v); draw(); });
  var L = DL.slider(ctrl, { label: 'Layers L', min: 1, max: 100, step: 1, value: 12 }, custom);
  var d = DL.slider(ctrl, { label: 'd_model', min: 64, max: 12288, step: 64, value: 768 }, custom);
  var ff = DL.slider(ctrl, { label: 'FFN multiplier', min: 1, max: 8, step: 0.5, value: 4 }, custom);
  var V = DL.slider(ctrl, { label: 'Vocab size', min: 1000, max: 200000, step: 1000, value: 50257, format: function (v) { return String(Math.round(v)); } }, custom);
  var T = DL.slider(ctrl, { label: 'Context T', min: 128, max: 32768, step: 128, value: 1024 }, draw);
  var tie = DL.toggle(ctrl, 'Tie input/output embeddings', true, custom);
  var learned = DL.toggle(ctrl, 'Learned position table', true, custom);
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Per layer: attention 4d² (+4d biases), FFN 2·m·d² (+(m+1)d biases), two LayerNorms 4d. Embeddings V·d (once if tied) plus T·d if positions are learned. Forward FLOPs per token ≈ 2 × (matmul parameters) + 4·L·T·d for attention scores and values (a multiply-add counts as 2 FLOPs); training ≈ 3× forward. Try the GPT-3 preset, then push context to 32768: the attention term takes over.' }));
  function apply(k) { var p = PRESETS[k]; if (!p) return; L.set(p.L); d.set(p.d); ff.set(p.ff); V.set(p.V); T.set(p.ctx); tie.set(p.tie); learned.set(p.learned); }
  function custom() { pre.set('custom'); draw(); }
  function human(x) { return x >= 1e9 ? DL.fmt(x / 1e9, 2) + 'B' : x >= 1e6 ? DL.fmt(x / 1e6, 1) + 'M' : DL.fmt(x / 1e3, 1) + 'K'; }
  function flops(x) { return x >= 1e12 ? DL.fmt(x / 1e12, 2) + ' TFLOPs' : x >= 1e9 ? DL.fmt(x / 1e9, 2) + ' GFLOPs' : DL.fmt(x / 1e6, 1) + ' MFLOPs'; }
  function draw() {
    var l = Math.round(L.get()), dm = Math.round(d.get()), m = ff.get(), v = Math.round(V.get()), t = Math.round(T.get());
    var attn = l * (4 * dm * dm + 4 * dm), ffn = l * (2 * m * dm * dm + (m + 1) * dm), ln = l * 4 * dm + 2 * dm;
    var emb = v * dm * (tie.get() ? 1 : 2), posp = learned.get() ? t * dm : 0, total = attn + ffn + ln + emb + posp;
    var nonEmb = attn + ffn;
    var ctx = c.ctx, col = DL.colors();
    ctx.clearRect(0, 0, c.w, c.h);
    var parts = [['attention', attn, col.blue], ['FFN', ffn, col.green], ['LayerNorm', ln, col.amber], ['embeddings', emb, col.purple], ['positions', posp, col.red]];
    var x = 20, y = 40, W = c.w - 40, hgt = 40;
    ctx.fillStyle = col.ink; ctx.font = '13px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('Parameter breakdown, total ' + human(total), x, 18);
    parts.forEach(function (p) { var w = W * p[1] / total; ctx.fillStyle = p[2]; ctx.fillRect(x, y, Math.max(0, w), hgt); x += w; });
    var ly = y + hgt + 22; x = 20;
    parts.forEach(function (p, i) {
      ctx.fillStyle = p[2]; ctx.fillRect(x, ly - 6, 12, 12); ctx.fillStyle = col.ink; ctx.font = '12px sans-serif';
      var s = p[0] + ' ' + human(p[1]) + ' (' + DL.fmt(100 * p[1] / total, 1) + '%)'; ctx.fillText(s, x + 17, ly);
      x += 30 + ctx.measureText(s).width; if (x > c.w - 190 && i < parts.length - 1) { x = 20; ly += 22; }
    });
    var by = ly + 36, att = 4 * l * t * dm, rest = 2 * nonEmb + 2 * v * dm, tot = att + rest;
    ctx.fillStyle = col.ink; ctx.font = '13px sans-serif'; ctx.fillText('Forward FLOPs per token at T = ' + t + ': dense matmuls vs attention scores', 20, by);
    ctx.fillStyle = col.green; ctx.fillRect(20, by + 14, W * rest / tot, 26); ctx.fillStyle = col.blue; ctx.fillRect(20 + W * rest / tot, by + 14, W * att / tot, 26);
    ctx.fillStyle = col.muted; ctx.font = '12px sans-serif'; ctx.fillText('dense ' + DL.fmt(100 * rest / tot, 0) + '%    attention scores ' + DL.fmt(100 * att / tot, 0) + '%', 20, by + 56);
    say('Total parameters <b>' + human(total) + '</b> (non-embedding <b>' + human(nonEmb) + '</b>). Forward ≈ <b>' + flops(tot) + '</b>/token, training ≈ <b>' + flops(3 * tot) + '</b>/token. Weights in bf16: <b>' + DL.fmt(2 * total / 1e9, 2) + ' GB</b>; mixed-precision Adam training state ≈ 16 bytes/param = <b>' + DL.fmt(16 * total / 1e9, 1) + ' GB</b>.');
  }
  apply('gpt2s');
  c.draw = draw;
  draw();
});
