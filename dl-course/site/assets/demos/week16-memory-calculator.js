DL.register('week16-memory-calculator', function (root) {
  // Per-GPU training memory for a GPT-style transformer: model states (ZeRO-aware) + activations.
  // Activation formula per layer (no tensor parallelism, 16-bit activations): s*b*h*(34 + 5*a*s/h) bytes
  // (Korthikanti et al., 2022). With checkpointing: keep s*b*h*2 per layer plus one full layer.
  var ctrl = DL.controls(root);
  var L = DL.slider(ctrl, { label: 'Layers L', min: 8, max: 96, step: 1, value: 32, format: function (v) { return String(Math.round(v)); } }, draw);
  var H = DL.slider(ctrl, { label: 'Hidden size h', min: 512, max: 12288, step: 256, value: 4096, format: function (v) { return String(Math.round(v)); } }, draw);
  var B = DL.slider(ctrl, { label: 'Micro-batch b', min: 1, max: 64, step: 1, value: 4, format: function (v) { return String(Math.round(v)); } }, draw);
  var S = DL.slider(ctrl, { label: 'Sequence s', min: 128, max: 16384, step: 128, value: 2048, format: function (v) { return String(Math.round(v)); } }, draw);
  var N = DL.slider(ctrl, { label: 'GPUs N', min: 1, max: 512, step: 1, value: 8, format: function (v) { return String(Math.round(v)); } }, draw);
  var prec = DL.select(ctrl, { label: 'Precision', value: 'amp', options: [{ value: 'fp32', label: 'FP32' }, { value: 'amp', label: 'Mixed (BF16/FP16 + FP32 master)' }] }, draw);
  var opt = DL.select(ctrl, { label: 'Optimizer', value: 'adam', options: [{ value: 'adam', label: 'Adam (m, v)' }, { value: 'sgdm', label: 'SGD + momentum' }] }, draw);
  var zero = DL.select(ctrl, { label: 'Sharding', value: '0', options: [{ value: '0', label: 'DDP (none)' }, { value: '1', label: 'ZeRO-1 (optimizer)' }, { value: '2', label: 'ZeRO-2 (+ gradients)' }, { value: '3', label: 'ZeRO-3 / FSDP (+ params)' }] }, draw);
  var ck = DL.toggle(ctrl, 'Gradient checkpointing', false, draw);
  var c = DL.canvas(root, { aspect: 0.4 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Bars show per-GPU memory in GB. Try: 7B-class model (L=32, h=4096) on 8 GPUs; compare DDP, ZeRO-1/2/3. Then raise the sequence length and see activations take over until you switch checkpointing on. Dashed lines mark 24, 40 and 80 GB cards. Estimates ignore fragmentation, CUDA context and temporary buffers (add 10-20%).' }));

  function compute() {
    var l = Math.round(L.get()), h = Math.round(H.get()), b = Math.round(B.get()), s = Math.round(S.get()), n = Math.round(N.get());
    var P = 12 * l * h * h + 50000 * h; // blocks + (tied) embedding with a 50k vocabulary
    var amp = prec.get() === 'amp', adam = opt.get() === 'adam', z = parseInt(zero.get(), 10);
    var pb = amp ? 2 : 4, gb = amp ? 2 : 4;
    var ob = (amp ? 4 : 0) + (adam ? 8 : 4); // master copy + m,v   (or momentum)
    var params = pb * P / (z >= 3 ? n : 1);
    var grads = gb * P / (z >= 2 ? n : 1);
    var optst = ob * P / (z >= 1 ? n : 1);
    var heads = Math.max(1, Math.round(h / 128));
    var perLayer = s * b * h * (34 + 5 * heads * s / h) * (amp ? 1 : 2);
    var act = ck.get() ? l * s * b * h * (amp ? 2 : 4) + perLayer : l * perLayer;
    return { P: P, params: params, grads: grads, opt: optst, act: act, total: params + grads + optst + act, n: n, ntok: b * s };
  }
  function draw() {
    if (!c.w) return;
    var d = compute(), ctx = c.ctx, col = DL.colors(), W = c.w, Ht = c.h, GB = 1e9;
    ctx.clearRect(0, 0, W, Ht);
    var segs = [['Parameters', d.params, col.blue], ['Gradients', d.grads, col.green], ['Optimizer states', d.opt, col.amber], ['Activations', d.act, col.purple]];
    var total = d.total / GB, xmax = Math.max(100, total * 1.08);
    var x0 = 20, x1 = W - 20, y = Ht * 0.25, bh = Ht * 0.28;
    function X(g) { return x0 + (x1 - x0) * g / xmax; }
    var acc = 0;
    ctx.font = '12px sans-serif'; ctx.textBaseline = 'middle';
    segs.forEach(function (sg) {
      var g = sg[1] / GB, w = X(acc + g) - X(acc);
      ctx.fillStyle = sg[2]; ctx.globalAlpha = 0.85; ctx.fillRect(X(acc), y, Math.max(0, w), bh); ctx.globalAlpha = 1;
      acc += g;
    });
    [24, 40, 80].forEach(function (g) {
      if (g > xmax) return;
      ctx.strokeStyle = g >= total ? col.green : col.red; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(X(g), y - 14); ctx.lineTo(X(g), y + bh + 14); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = col.muted; ctx.textAlign = 'center'; ctx.fillText(g + ' GB', X(g), y - 22);
    });
    // legend with numbers
    var ly = y + bh + 38; ctx.textAlign = 'left';
    segs.forEach(function (sg, i) {
      var lx = 20 + (i % 2) * (W / 2), yy = ly + Math.floor(i / 2) * 22;
      ctx.fillStyle = sg[2]; ctx.fillRect(lx, yy - 6, 12, 12);
      ctx.fillStyle = col.ink; ctx.fillText(sg[0] + ': ' + DL.fmt(sg[1] / GB, 1) + ' GB', lx + 18, yy);
    });
    ctx.fillStyle = col.ink; ctx.font = 'bold 13px sans-serif';
    ctx.fillText('Total per GPU: ' + DL.fmt(total, 1) + ' GB', 20, Ht - 18);
    var fits = total <= 80 ? (total <= 40 ? (total <= 24 ? 'fits a 24 GB card' : 'fits a 40 GB card') : 'fits an 80 GB card') : 'does not fit an 80 GB card';
    say('parameters: <b>' + DL.fmt(d.P / 1e9, 2) + ' B</b> &nbsp; tokens per micro-batch: ' + d.ntok + ' &nbsp; model states: <b>' + DL.fmt((d.params + d.grads + d.opt) / GB, 1) + ' GB</b> &nbsp; activations: <b>' + DL.fmt(d.act / GB, 1) + ' GB</b> &nbsp; → ' + fits);
  }
  c.draw = draw;
  draw();
});
