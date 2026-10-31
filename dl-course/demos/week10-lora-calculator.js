DL.register('week10-lora-calculator', function (root) {
  var models = {
    'opt125': { name: 'OPT-125M (12 layers, d=768)', L: 12, d: 768, ff: 3072, kv: 768, gated: false, total: 125.2e6 },
    'llama7': { name: 'LLaMA-7B shape (32 layers, d=4096)', L: 32, d: 4096, ff: 11008, kv: 4096, gated: true, total: 6.74e9 },
    'llama13': { name: 'LLaMA-13B shape (40 layers, d=5120)', L: 40, d: 5120, ff: 13824, kv: 5120, gated: true, total: 13.0e9 },
    'llama70': { name: 'Llama-70B shape (80 layers, d=8192, GQA 8 kv heads)', L: 80, d: 8192, ff: 28672, kv: 1024, gated: true, total: 69.0e9 }
  };
  var ctrl = DL.controls(root);
  var sel = DL.select(ctrl, { label: 'Model', value: 'llama7', options: Object.keys(models).map(function (k) { return { value: k, label: models[k].name }; }) }, draw);
  var rk = DL.slider(ctrl, { label: 'Rank r (log2)', min: 0, max: 8, step: 1, value: 3, format: function (v) { return String(Math.pow(2, Math.round(v))); } }, draw);
  var row = DL.row(ctrl);
  var tq = DL.toggle(row, 'W_q', true, draw), tk = DL.toggle(row, 'W_k', false, draw), tv = DL.toggle(row, 'W_v', true, draw), to = DL.toggle(row, 'W_o', false, draw), tf = DL.toggle(row, 'FFN (all)', false, draw);
  var bits = DL.select(ctrl, { label: 'Frozen base precision', value: '16', options: [{ value: '16', label: '16-bit (BF16)' }, { value: '8', label: '8-bit' }, { value: '4.13', label: '4-bit NF4 (+ double quant)' }] }, draw);
  var c = DL.canvas(root, { aspect: 0.42 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Per adapted matrix LoRA trains r(d_in + d_out) parameters instead of d_in d_out. Memory bars count weights and optimizer state only (no activations): full fine-tuning with mixed-precision AdamW is 16 bytes/param; LoRA keeps the frozen base at the chosen precision and trains adapters with 16 bytes/param. Try r = 8 on W_q, W_v, then add the FFN and watch the count and the full-vs-LoRA memory gap.' }));
  function human(x) { return x >= 1e9 ? (x / 1e9).toFixed(2) + 'B' : x >= 1e6 ? (x / 1e6).toFixed(2) + 'M' : x >= 1e3 ? (x / 1e3).toFixed(1) + 'K' : String(Math.round(x)); }
  function gb(x) { return (x / 1e9).toFixed(1) + ' GB'; }
  function draw() {
    var m = models[sel.get()], r = Math.pow(2, Math.round(rk.get())), d = m.d, per = 0;
    if (tq.get()) per += r * (d + d);
    if (tk.get()) per += r * (d + m.kv);
    if (tv.get()) per += r * (d + m.kv);
    if (to.get()) per += r * (d + d);
    if (tf.get()) per += (m.gated ? 3 : 2) * r * (d + m.ff);
    var trainable = per * m.L, frac = trainable / m.total;
    var full = 16 * m.total, bb = parseFloat(bits.get());
    var lora = m.total * bb / 8 + 16 * trainable;
    var ctx = c.ctx, col = DL.colors();
    ctx.clearRect(0, 0, c.w, c.h);
    var items = [['Full fine-tune (16 B/param)', full, col.red], ['LoRA, base at ' + (bb > 8 ? '4-bit' : bb + '-bit'), lora, col.accent], ['Trainable state only', 16 * trainable, col.green]];
    var Lm = 190, Rm = 90, top = 22, bh = Math.min(34, (c.h - top - 10) / 3 - 10), mx = full;
    ctx.font = '12px ' + getComputedStyle(document.body).getPropertyValue('--font-sans');
    items.forEach(function (it, i) {
      var y = top + i * (bh + 14), w = Math.max(2, (c.w - Lm - Rm) * it[1] / mx);
      ctx.fillStyle = col.ink; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(it[0], Lm - 8, y + bh / 2);
      ctx.fillStyle = it[2]; ctx.globalAlpha = 0.85; ctx.fillRect(Lm, y, w, bh); ctx.globalAlpha = 1;
      ctx.fillStyle = col.ink; ctx.textAlign = 'left'; ctx.fillText(gb(it[1]), Lm + w + 6, y + bh / 2);
    });
    ctx.fillStyle = col.muted; ctx.textAlign = 'left'; ctx.fillText('Training memory for weights + gradients + optimizer state (log-free linear scale)', Lm, 10);
    say('Adapted matrices per layer give <b>' + human(per) + '</b> params/layer; total trainable <b>' + human(trainable) + '</b> = <b>' + (100 * frac).toFixed(3) + '%</b> of ' + human(m.total) + '.<br>' +
      'Adapter file at 16-bit: about <b>' + (trainable * 2 / 1e6).toFixed(1) + ' MB</b>. Rank r = ' + r + (per === 0 ? ' (select at least one matrix)' : '') + '.');
  }
  c.draw = draw;
});
