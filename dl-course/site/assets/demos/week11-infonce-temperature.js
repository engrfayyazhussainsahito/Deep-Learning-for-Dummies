DL.register('week11-infonce-temperature', function (root) {
  var ctrl = DL.controls(root);
  var tauS = DL.slider(ctrl, { label: 'Temperature τ', min: -2, max: 0, step: 0.01, value: -0.85, format: function (v) { return DL.fmt(Math.pow(10, v), 3); } }, draw);
  var posS = DL.slider(ctrl, { label: 'Positive similarity s⁺', min: -0.2, max: 1, step: 0.01, value: 0.7 }, draw);
  var negS = DL.slider(ctrl, { label: 'Mean negative similarity', min: -0.5, max: 0.9, step: 0.01, value: 0.1 }, draw);
  var spreadS = DL.slider(ctrl, { label: 'Negative spread', min: 0, max: 0.5, step: 0.01, value: 0.25 }, draw);
  var kS = DL.slider(ctrl, { label: 'Negatives K', min: 2, max: 63, step: 1, value: 15 }, draw);
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Left: softmax probability assigned to the positive (first, red) and to each negative (blue, sorted by similarity). The gradient pushing a negative away is proportional to its probability, so small τ concentrates the push on the hardest negatives. Right: the loss as a function of τ for the current similarities. Try τ = 0.05 with hard negatives (spread 0.5, mean 0.5) versus τ = 1.' }));
  var base = DL.rng(11), z = []; for (var q = 0; q < 63; q++) z.push(DL.randn(base));
  function negs() {
    var K = Math.round(kS.get()), a = [];
    for (var i = 0; i < K; i++) a.push(DL.clamp(negS.get() + spreadS.get() * z[i], -1, 1));
    a.sort(function (x, y) { return y - x; });
    return a;
  }
  function probs(sp, ns, tau) { var l = [sp / tau]; ns.forEach(function (v) { l.push(v / tau); }); return DL.softmax(l); }
  function loss(sp, ns, tau) { var p = probs(sp, ns, tau); return -Math.log(Math.max(p[0], 1e-300)); }
  function draw() {
    if (!c.w) return;
    var ctx = c.ctx, col = DL.colors(), tau = Math.pow(10, tauS.get()), sp = posS.get(), ns = negs();
    var p = probs(sp, ns, tau), L = -Math.log(Math.max(p[0], 1e-300));
    ctx.clearRect(0, 0, c.w, c.h);
    var mono = getComputedStyle(document.body).getPropertyValue('--font-mono');
    var lw = c.w * 0.5, l = 40, t = 24, b = c.h - 34, n = ns.length + 1, bw = Math.max(1.5, (lw - l - 6) / n - 2), i;
    var pm = Math.max(0.05, Math.max.apply(null, p));
    ctx.save(); ctx.strokeStyle = col.muted; ctx.beginPath(); ctx.moveTo(l, t); ctx.lineTo(l, b); ctx.lineTo(lw, b); ctx.stroke();
    ctx.fillStyle = col.muted; ctx.font = '11px ' + mono; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    [0, 0.5, 1].forEach(function (f) { var y = b - f * (b - t); ctx.fillText((f * pm).toFixed(2), l - 4, y); ctx.strokeStyle = col.line; ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(lw, y); ctx.stroke(); });
    for (i = 0; i < n; i++) { var h = p[i] / pm * (b - t), x = l + 4 + i * (bw + 2); ctx.fillStyle = i === 0 ? col.red : col.blue; ctx.fillRect(x, b - h, bw, h); }
    ctx.fillStyle = col.ink; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.font = '12px ' + getComputedStyle(document.body).getPropertyValue('--font-sans');
    ctx.fillText('softmax probabilities (red = positive)', l, 4);
    ctx.restore();
    var P = DL.plot(c, { xmin: -2, xmax: 0, ymin: 0, ymax: Math.log(n) * 1.15 + 0.1, xlabel: 'log10 τ', ylabel: 'InfoNCE loss', pad: { l: lw + 52, r: 12, t: 24, b: 34 } });
    P.grid(); P.axes();
    P.fn(function (x) { return loss(sp, ns, Math.pow(10, x)); }, col.accent, 2.5);
    P.hline(Math.log(n), col.muted);
    P.points([tauS.get()], [L], col.ink, 5);
    P.text('log(K+1)', -1.95, Math.log(n) + 0.12, col.muted, 'left');
    var hard = ns[0], gHard = p[1] / tau, gPos = (1 - p[0]) / tau;
    var top3 = p[1] + (p[2] || 0) + (p[3] || 0);
    say('τ = <b>' + DL.fmt(tau, 3) + '</b> &nbsp; loss = <b>' + DL.fmt(L, 3) + '</b> &nbsp; P(positive) = <b>' + DL.fmt(p[0], 3) + '</b> &nbsp; (chance ' + DL.fmt(1 / n, 3) + ')<br>' +
      '|∂L/∂s| hardest negative (sim ' + DL.fmt(hard, 2) + '): <b>' + DL.fmt(gHard, 3) + '</b> &nbsp; positive: <b>' + DL.fmt(gPos, 3) + '</b> &nbsp; share of the negative push carried by the top-3 negatives: <b>' + DL.fmt(100 * top3 / Math.max(1e-12, 1 - p[0]), 1) + '%</b>');
  }
  c.draw = draw;
});
