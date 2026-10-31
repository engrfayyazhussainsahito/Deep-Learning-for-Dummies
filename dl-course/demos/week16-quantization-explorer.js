DL.register('week16-quantization-explorer', function (root) {
  // A synthetic weight matrix of 8 output channels x 32 weights with different channel scales and an optional outlier.
  var ctrl = DL.controls(root);
  var bits = DL.slider(ctrl, { label: 'Bits', min: 2, max: 8, step: 1, value: 4, format: function (v) { return String(Math.round(v)); } }, draw);
  var gran = DL.select(ctrl, { label: 'Scale granularity', value: 'tensor', options: [{ value: 'tensor', label: 'Per-tensor' }, { value: 'channel', label: 'Per-channel' }, { value: 'group', label: 'Group of 8' }] }, draw);
  var sym = DL.select(ctrl, { label: 'Range', value: 'sym', options: [{ value: 'sym', label: 'Symmetric (zero-point 0)' }, { value: 'asym', label: 'Asymmetric (min/max)' }] }, draw);
  var out = DL.slider(ctrl, { label: 'Outlier size', min: 0, max: 30, step: 1, value: 0, format: function (v) { return v === 0 ? 'none' : v + '×σ'; } }, draw);
  var seedB = DL.button(ctrl, 'New weights', function () { seed++; build(); draw(); });
  var C = 8, G = 32, seed = 3, W0;
  function build() {
    var rg = DL.rng(seed), r = function () { return DL.randn(rg); }; W0 = [];
    for (var i = 0; i < C; i++) { var sc = Math.exp(0.7 * r()) * 0.5, row = []; for (var j = 0; j < G; j++) row.push(sc * r()); W0.push(row); }
  }
  build();
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Top: all 256 weights in order (grey = original, coloured dots = dequantized). Bottom: relative RMS error per output channel. Add an outlier in channel 2 and watch per-tensor error explode for every channel while per-channel contains the damage to one. Drop to 3 bits to see why 4-bit methods need groups and smarter rounding (GPTQ, AWQ).' }));

  function quant(vals, nb, asym) {
    var mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals), lo, hi, s, z;
    if (asym) { lo = 0; hi = Math.pow(2, nb) - 1; s = (mx - mn) / (hi - lo) || 1e-12; z = Math.round(lo - mn / s); return vals.map(function (v) { var q = DL.clamp(Math.round(v / s) + z, lo, hi); return s * (q - z); }); }
    var qm = Math.pow(2, nb - 1) - 1; s = Math.max(Math.abs(mn), Math.abs(mx)) / qm || 1e-12;
    return vals.map(function (v) { return s * DL.clamp(Math.round(v / s), -qm, qm); });
  }
  function run() {
    var nb = Math.round(bits.get()), asym = sym.get() === 'asym', g = gran.get();
    var W = W0.map(function (r) { return r.slice(); });
    if (out.get() > 0) { var sd = 0.5; W[2][5] = out.get() * sd; }
    var Q = W.map(function (r) { return r.slice(); });
    if (g === 'tensor') { var flat = [].concat.apply([], W), qf = quant(flat, nb, asym); for (var i = 0; i < C; i++) for (var j = 0; j < G; j++) Q[i][j] = qf[i * G + j]; }
    else if (g === 'channel') { for (var i2 = 0; i2 < C; i2++) Q[i2] = quant(W[i2], nb, asym); }
    else { for (var i3 = 0; i3 < C; i3++) for (var k = 0; k < G; k += 8) { var q = quant(W[i3].slice(k, k + 8), nb, asym); for (var m = 0; m < 8; m++) Q[i3][k + m] = q[m]; } }
    return { W: W, Q: Q };
  }
  function draw() {
    if (!c.w) return;
    var r = run(), ctx = c.ctx, col = DL.colors(), Wd = c.w, H = c.h;
    ctx.clearRect(0, 0, Wd, H);
    var all = [].concat.apply([], r.W), allq = [].concat.apply([], r.Q), n = all.length;
    var amax = Math.max.apply(null, all.map(Math.abs)) * 1.1 || 1;
    var P = DL.plot({ ctx: ctx, w: Wd, h: H * 0.58 }, { xmin: 0, xmax: n, ymin: -amax, ymax: amax, xlabel: 'weight index (8 channels × 32)', ylabel: 'value', pad: { l: 44, r: 10, t: 8, b: 30 } });
    P.grid(); P.axes();
    for (var i = 1; i < C; i++) P.vline(i * G, col.line);
    P.line(DL.range(n), all, col.muted, 1.2);
    P.points(DL.range(n), allq, col.accent, 2.5);
    // per-channel errors
    var errs = [], se = 0, sw = 0;
    for (var a = 0; a < C; a++) { var e = 0, w2 = 0; for (var j = 0; j < G; j++) { var d = r.W[a][j] - r.Q[a][j]; e += d * d; w2 += r.W[a][j] * r.W[a][j]; } errs.push(Math.sqrt(e / (w2 || 1))); se += e; sw += w2; }
    var top = H * 0.62, bh = H * 0.28, emax = Math.max(0.05, Math.max.apply(null, errs) * 1.15), bw = (Wd - 64) / C;
    ctx.font = '11px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (var k2 = 0; k2 < C; k2++) {
      var hh = bh * errs[k2] / emax;
      ctx.fillStyle = k2 === 2 && out.get() > 0 ? col.red : col.blue; ctx.globalAlpha = 0.8;
      ctx.fillRect(44 + k2 * bw + 6, top + bh - hh, bw - 12, hh); ctx.globalAlpha = 1;
      ctx.fillStyle = col.muted; ctx.fillText('ch ' + k2, 44 + k2 * bw + bw / 2, top + bh + 4);
      ctx.fillText(DL.fmt(errs[k2], 2), 44 + k2 * bw + bw / 2, top + bh - hh - 14);
    }
    ctx.save(); ctx.translate(12, top + bh / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillStyle = col.muted; ctx.fillText('rel. RMS error', 0, 0); ctx.restore();
    var rel = Math.sqrt(se / (sw || 1)), sqnr = 10 * Math.log10(sw / (se || 1e-30));
    var nb = Math.round(bits.get());
    say('relative RMS error: <b>' + DL.fmt(rel, 3) + '</b> &nbsp; SQNR: <b>' + DL.fmt(sqnr, 1) + ' dB</b> &nbsp; levels: ' + Math.pow(2, nb) + ' &nbsp; (rule of thumb: +6 dB per extra bit when the range is used well)');
  }
  c.draw = draw;
  draw();
});
