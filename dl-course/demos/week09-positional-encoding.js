DL.register('week09-positional-encoding', function (root) {
  var ctrl = DL.controls(root);
  var mode = DL.select(ctrl, { label: 'View', value: 'sin', options: [
    { value: 'sin', label: 'Sinusoidal PE matrix (positions × dims)' },
    { value: 'dot', label: 'Similarity PE(p)·PE(q) vs offset' },
    { value: 'rope', label: 'RoPE: score vs relative distance' },
    { value: 'alibi', label: 'ALiBi: linear distance penalty' }] }, draw);
  var dm = DL.slider(ctrl, { label: 'd_model', min: 8, max: 128, step: 8, value: 64 }, draw);
  var base = DL.slider(ctrl, { label: 'Base (10000 in paper)', min: 100, max: 100000, step: 100, value: 10000, format: function (v) { return String(Math.round(v)); } }, draw);
  var p0 = DL.slider(ctrl, { label: 'Reference position p', min: 0, max: 100, step: 1, value: 40 }, draw);
  var c = DL.canvas(root, { aspect: 0.55 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Matrix view: columns are dimensions; left columns oscillate fast, right columns slowly (a bank of clocks with geometric periods). Similarity view: the dot product of PE(p) and PE(p+k) depends only on the offset k, decays with |k| and is symmetric. RoPE view: with q = k = all ones, the rotated dot product is the mean of cos(kθ_i): a purely relative signal. Lower the base to make long-range positions alias sooner; ALiBi shows per-head slopes 2^(-8h/H).' }));
  var NPOS = 100;
  function pe(p, d, b) {
    var v = [], i;
    for (i = 0; i < d; i += 2) { var w = Math.pow(b, -i / d); v.push(Math.sin(p * w), Math.cos(p * w)); }
    return v;
  }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function draw() {
    var ctx = c.ctx, col = DL.colors(), d = Math.round(dm.get()), b = base.get(), m = mode.get(), p = Math.round(p0.get());
    ctx.clearRect(0, 0, c.w, c.h);
    if (m === 'sin') {
      var M = [], i;
      for (i = 0; i < NPOS; i++) {
        var v = pe(i, d, b), r = [];
        for (var k = 0; k < d; k += 2) { r.push(v[k]); }
        // reorder: all sin dims then cos dims, as in many implementations
        var r2 = []; for (k = 0; k < d; k += 2) r2.push(v[k]); for (k = 0; k < d; k += 2) r2.push(v[k + 1]);
        M.push(r2);
      }
      var x0 = 50, y0 = 16, w = c.w - x0 - 16, h = c.h - y0 - 34;
      DL.heatmap(ctx, M, x0, y0, w, h, { min: -1, max: 1, cmap: DL.cmap.diverge });
      ctx.fillStyle = col.muted; ctx.font = '11px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText('dimension (sin block then cos block; frequency decreases within each block)', x0 + w / 2, y0 + h + 6);
      ctx.save(); ctx.translate(12, y0 + h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('position 0 … ' + (NPOS - 1), 0, 0); ctx.restore();
      ctx.strokeStyle = col.accent; ctx.lineWidth = 1.5; var yy = y0 + (p + 0.5) * h / NPOS; ctx.beginPath(); ctx.moveTo(x0, yy); ctx.lineTo(x0 + w, yy); ctx.stroke();
      var per = []; for (i = 0; i < d; i += 2) per.push(2 * Math.PI * Math.pow(b, i / d));
      say('Wavelengths run from <b>2π ≈ 6.3</b> positions (fastest pair) to <b>' + DL.fmt(per[per.length - 1], 0) + '</b> positions (slowest pair). Highlighted row: position ' + p + '.');
    } else {
      var P = DL.plot(c, { xmin: -NPOS + 1, xmax: NPOS - 1, ymin: m === 'alibi' ? -12 : -0.5, ymax: m === 'alibi' ? 0.5 : 1.05, xlabel: 'offset k = n − m', ylabel: m === 'alibi' ? 'bias added to score' : 'normalized score' });
      P.clear(); P.grid(); P.axes();
      var xs = DL.range(2 * NPOS - 1).map(function (j) { return j - NPOS + 1; });
      if (m === 'dot') {
        var ref = pe(p, d, b), nn = dot(ref, ref), ys = xs.map(function (k) { var q = p + k; return q < 0 ? NaN : dot(ref, pe(q, d, b)) / nn; });
        var xs2 = [], ys2 = []; xs.forEach(function (k, j) { if (!isNaN(ys[j])) { xs2.push(k); ys2.push(ys[j]); } });
        P.line(xs2, ys2, col.accent, 2.2);
        say('PE(p)·PE(p+k) / |PE(p)|² for p = <b>' + p + '</b>. Equals (2/d)·Σ cos(kω_i) for any p: the shape is the same wherever you stand, only truncated at position 0.');
      } else if (m === 'rope') {
        var th = []; for (var i2 = 0; i2 < d / 2; i2++) th.push(Math.pow(b, -2 * i2 / d));
        var yr = xs.map(function (k) { var s = 0; th.forEach(function (t) { s += Math.cos(k * t); }); return s / th.length; });
        P.line(xs, yr, col.blue, 2.2); P.hline(0, col.muted);
        say('RoPE score for q = k = 1 (all ones) is mean<sub>i</sub> cos(k θ<sub>i</sub>) with θ<sub>i</sub> = base<sup>−2i/d</sup>. It depends only on the offset k = n − m; real models learn q,k so each frequency band carries its own signal.');
      } else {
        var cols = [col.blue, col.green, col.amber, col.purple, col.red, col.teal, col.accent, col.muted], H = 8, h2;
        for (h2 = 1; h2 <= H; h2++) { var sl = Math.pow(2, -8 * h2 / H), xe = Math.min(NPOS - 1, 12 / sl); P.line([-xe, 0], [-sl * xe, 0], cols[h2 - 1], 1.8); P.line([0, xe], [0, -sl * xe], cols[h2 - 1], 1.8); }
        say('ALiBi with 8 heads: slopes 2<sup>−1</sup>, 2<sup>−2</sup>, …, 2<sup>−8</sup>. Bias = −slope·|k| is added to the logits before the softmax; steep heads are local, shallow heads see far. (The plot clips below −12; in a causal model only k ≤ 0 is used.)');
      }
    }
  }
  c.draw = draw;
  draw();
});
