DL.register('week16-roofline-explorer', function (root) {
  // Roofline for a matrix multiply C = A B, A:(M,K) B:(K,N), counting only compulsory traffic (read A, B; write C).
  var HW = { a100: { n: 'A100 (BF16 dense)', P: 312, B: 2.039 }, h100: { n: 'H100 SXM (BF16 dense)', P: 989, B: 3.35 }, v100: { n: 'V100 (FP16 tensor)', P: 125, B: 0.9 }, cpu: { n: 'Server CPU socket (FP32, illustrative)', P: 2, B: 0.2 } };
  var ctrl = DL.controls(root);
  var hw = DL.select(ctrl, { label: 'Hardware', value: 'a100', options: Object.keys(HW).map(function (k) { return { value: k, label: HW[k].n }; }) }, draw);
  function p2(v) { return String(Math.round(Math.pow(2, v))); }
  var M = DL.slider(ctrl, { label: 'M (rows of A)', min: 0, max: 14, step: 1, value: 3, format: p2 }, draw);
  var K = DL.slider(ctrl, { label: 'K (inner dim)', min: 4, max: 14, step: 1, value: 12, format: p2 }, draw);
  var N = DL.slider(ctrl, { label: 'N (cols of B)', min: 4, max: 14, step: 1, value: 12, format: p2 }, draw);
  var by = DL.select(ctrl, { label: 'Bytes/element', value: '2', options: [{ value: '4', label: '4 (FP32)' }, { value: '2', label: '2 (FP16/BF16)' }, { value: '1', label: '1 (INT8/FP8)' }, { value: '0.5', label: '0.5 (INT4)' }] }, draw);
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'M = 1 to 16 is autoregressive decoding or a tiny batch: the weight matrix (K×N) is read once per few FLOPs, so you sit far left on the bandwidth slope, and halving bytes per weight (quantization) nearly doubles speed. M = 4096 with big K, N is training or prefill: compute-bound, where only faster math helps.' }));

  function draw() {
    if (!c.w) return;
    var h = HW[hw.get()], m = Math.pow(2, Math.round(M.get())), k = Math.pow(2, Math.round(K.get())), n = Math.pow(2, Math.round(N.get())), b = parseFloat(by.get());
    var flops = 2 * m * n * k, bytes = b * (m * k + k * n + m * n), I = flops / bytes;
    var ridge = h.P / h.B; // FLOP/byte  (TFLOP/s over TB/s)
    var att = Math.min(h.P, h.B * I);
    var ctx = c.ctx, col = DL.colors(), W = c.w, H = c.h, pl = 56, pr = 14, pt = 14, pb = 36;
    ctx.clearRect(0, 0, W, H);
    var x0 = -1, x1 = 4, ymin = Math.log10(h.P) - 4, ymax = Math.log10(h.P) + 0.5;
    function X(v) { return pl + (Math.log10(v) - x0) / (x1 - x0) * (W - pl - pr); }
    function Y(v) { return H - pb - (Math.log10(Math.max(v, 1e-9)) - ymin) / (ymax - ymin) * (H - pt - pb); }
    ctx.strokeStyle = col.line; ctx.fillStyle = col.muted; ctx.font = '11px sans-serif'; ctx.lineWidth = 1;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (var e = x0; e <= x1; e++) { ctx.beginPath(); ctx.moveTo(X(Math.pow(10, e)), pt); ctx.lineTo(X(Math.pow(10, e)), H - pb); ctx.stroke(); ctx.fillText(String(Math.pow(10, e)), X(Math.pow(10, e)), H - pb + 4); }
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (var ye = Math.ceil(ymin); ye <= Math.floor(ymax); ye++) { ctx.beginPath(); ctx.moveTo(pl, Y(Math.pow(10, ye))); ctx.lineTo(W - pr, Y(Math.pow(10, ye))); ctx.stroke(); ctx.fillText(ye >= 0 ? String(Math.pow(10, ye)) : Math.pow(10, ye).toFixed(-ye), pl - 4, Y(Math.pow(10, ye))); }
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText('arithmetic intensity (FLOP/byte)', (pl + W - pr) / 2, H - 1);
    ctx.save(); ctx.translate(12, (pt + H - pb) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillText('TFLOP/s', 0, 0); ctx.restore();
    // roof
    ctx.strokeStyle = col.accent; ctx.lineWidth = 3; ctx.beginPath();
    ctx.moveTo(X(Math.pow(10, x0)), Y(h.B * Math.pow(10, x0))); ctx.lineTo(X(ridge), Y(h.P)); ctx.lineTo(X(Math.pow(10, x1)), Y(h.P)); ctx.stroke();
    ctx.setLineDash([5, 4]); ctx.strokeStyle = col.muted; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(X(ridge), Y(h.P)); ctx.lineTo(X(ridge), H - pb); ctx.stroke(); ctx.setLineDash([]);
    var Ic = DL.clamp(I, Math.pow(10, x0), Math.pow(10, x1));
    ctx.fillStyle = I < ridge ? col.red : col.green; ctx.beginPath(); ctx.arc(X(Ic), Y(att), 7, 0, 7); ctx.fill();
    ctx.fillStyle = col.ink; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText('ridge ' + DL.fmt(ridge, 0) + ' FLOP/B', X(ridge) + 4, H - pb - 4);
    var t_comp = flops / (h.P * 1e12), t_mem = bytes / (h.B * 1e12);
    say('FLOPs: <b>' + flops.toExponential(2) + '</b> &nbsp; bytes: <b>' + bytes.toExponential(2) + '</b> &nbsp; intensity I = <b>' + DL.fmt(I, 1) + '</b> FLOP/B &nbsp; → <b>' + (I < ridge ? 'memory-bound' : 'compute-bound') + '</b>, attainable ' + DL.fmt(att, 1) + ' TFLOP/s (' + DL.fmt(100 * att / h.P, 1) + '% of peak) &nbsp; time ≥ ' + DL.fmt(Math.max(t_comp, t_mem) * 1e6, 2) + ' µs');
  }
  c.draw = draw;
  draw();
});
