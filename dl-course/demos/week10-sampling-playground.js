DL.register('week10-sampling-playground', function (root) {
  var words = ['mat', 'floor', 'sofa', 'couch', 'chair', 'bed', 'roof', 'table', 'moon', 'idea', 'banana', 'quantum'];
  var logits = [4.0, 3.4, 2.9, 2.2, 1.5, 1.0, 0.3, -0.2, -0.8, -1.5, -3.0, -4.0];
  var rand = DL.rng(7), counts = DL.range(words.length).map(function () { return 0; }), total = 0, last = '-';
  var ctrl = DL.controls(root);
  var T = DL.slider(ctrl, { label: 'Temperature T', min: 0.05, max: 3, step: 0.05, value: 1, format: function (v) { return (+v).toFixed(2); } }, draw);
  var K = DL.slider(ctrl, { label: 'Top-k (0 = off)', min: 0, max: 12, step: 1, value: 0, format: function (v) { return v == 0 ? 'off' : String(Math.round(v)); } }, draw);
  var Pp = DL.slider(ctrl, { label: 'Top-p', min: 0.05, max: 1, step: 0.01, value: 1, format: function (v) { return (+v).toFixed(2); } }, draw);
  var row = DL.row(root);
  DL.button(row, 'Sample 1', function () { drawSamples(1); });
  DL.button(row, 'Sample 200', function () { drawSamples(200); });
  DL.button(row, 'Reset counts', function () { counts = counts.map(function () { return 0; }); total = 0; last = '-'; draw(); });
  var c = DL.canvas(root, { aspect: 0.55 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Hint: logits are fixed (next word after "The cat sat on the ..."). Lower T sharpens the distribution; top-k keeps a fixed number of words; top-p keeps the smallest set whose probability reaches p. Order of operations: temperature, then top-k, then top-p, then renormalize. Draw samples and compare the empirical bars with the exact ones.' }));

  function compute() {
    var t = Math.max(0.01, T.get()), k = Math.round(K.get()), p = Pp.get(), n = words.length;
    var base = DL.softmax(logits, 1), prob = DL.softmax(logits, t);
    var order = DL.range(n).sort(function (a, b) { return prob[b] - prob[a]; });
    var keep = prob.map(function () { return true; }), i;
    if (k > 0) for (i = k; i < n; i++) keep[order[i]] = false;
    if (p < 1) {
      var cum = 0;
      for (i = 0; i < n; i++) { if (cum >= p - 1e-12) keep[order[i]] = false; cum += prob[order[i]]; }
    }
    var z = 0; for (i = 0; i < n; i++) if (keep[i]) z += prob[i];
    var q = prob.map(function (v, j) { return keep[j] ? v / z : 0; });
    return { base: base, prob: prob, q: q, keep: keep };
  }
  function drawSamples(m) {
    var s = compute(), i, j;
    for (i = 0; i < m; i++) {
      var u = rand(), acc = 0, pick = 0;
      for (j = 0; j < s.q.length; j++) { acc += s.q[j]; if (u <= acc) { pick = j; break; } pick = j; }
      counts[pick]++; total++; last = words[pick];
    }
    draw();
  }
  function draw() {
    var ctx = c.ctx, col = DL.colors(), s = compute(), n = words.length, i;
    ctx.clearRect(0, 0, c.w, c.h);
    var L = 40, R = 10, Tp = 16, B = 52, W = c.w - L - R, H = c.h - Tp - B, bw = W / n;
    var ymax = Math.max(0.1, Math.max.apply(null, s.q), Math.max.apply(null, s.base));
    ymax = Math.ceil(ymax * 10) / 10;
    ctx.font = '11px ' + getComputedStyle(document.body).getPropertyValue('--font-sans');
    ctx.strokeStyle = col.line; ctx.fillStyle = col.muted; ctx.lineWidth = 1;
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (i = 0; i <= 4; i++) {
      var yv = ymax * i / 4, y = Tp + H - H * i / 4;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + W, y); ctx.globalAlpha = 0.5; ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillText(yv.toFixed(2), L - 5, y);
    }
    for (i = 0; i < n; i++) {
      var x = L + i * bw, h1 = H * s.q[i] / ymax, hb = H * s.base[i] / ymax;
      ctx.fillStyle = s.keep[i] ? col.accent : col.soft; ctx.globalAlpha = s.keep[i] ? 0.85 : 1;
      ctx.fillRect(x + bw * 0.14, Tp + H - h1, bw * 0.72, Math.max(h1, 0));
      ctx.globalAlpha = 1;
      ctx.strokeStyle = col.muted; ctx.setLineDash([3, 3]); ctx.lineWidth = 1.2;
      ctx.strokeRect(x + bw * 0.14, Tp + H - hb, bw * 0.72, Math.max(hb, 0)); ctx.setLineDash([]);
      if (total > 0) {
        var e = counts[i] / total, ye = Tp + H - H * e / ymax;
        ctx.fillStyle = col.amber; ctx.beginPath(); ctx.arc(x + bw / 2, ye, 4, 0, 6.2832); ctx.fill();
      }
      ctx.fillStyle = s.keep[i] ? col.ink : col.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.save(); ctx.translate(x + bw / 2, Tp + H + 6); ctx.rotate(0.6); ctx.textAlign = 'left'; ctx.fillText(words[i], 0, 0); ctx.restore();
    }
    ctx.fillStyle = col.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('filled: sampling distribution   dashed: original (T = 1)   dots: empirical frequency', L, c.h - 8);
    var H0 = 0, Hq = 0, kept = 0;
    for (i = 0; i < n; i++) { if (s.base[i] > 0) H0 -= s.base[i] * Math.log(s.base[i]); if (s.q[i] > 0) Hq -= s.q[i] * Math.log(s.q[i]); if (s.keep[i]) kept++; }
    say('kept tokens: <b>' + kept + '</b> of ' + n + ' &nbsp; top-1 prob: <b>' + DL.fmt(Math.max.apply(null, s.q), 3) + '</b> &nbsp; entropy: <b>' + DL.fmt(Hq, 3) + '</b> nats (original ' + DL.fmt(H0, 3) + ') &nbsp; effective choices exp(H): <b>' + DL.fmt(Math.exp(Hq), 2) + '</b><br>samples drawn: ' + total + ' &nbsp; last: <b>' + last + '</b>');
  }
  c.draw = draw;
});
