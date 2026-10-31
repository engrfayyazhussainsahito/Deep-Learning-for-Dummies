DL.register('week08-analogy-explorer', function (root) {
  // A tiny HAND-BUILT 2D "embedding" (illustrative, not learned): axis 1 mixes gender and geography, axis 2 mixes royalty and capital status.
  var base = {
    man: [-1.0, 0.0], woman: [1.0, 0.0], boy: [-1.0, -0.55], girl: [1.0, -0.5],
    prince: [-1.0, 0.65], princess: [1.0, 0.6], king: [-1.0, 1.25], queen: [1.0, 1.15],
    france: [2.8, 2.0], paris: [2.95, 1.05], germany: [4.0, 2.3], berlin: [4.1, 1.3],
    italy: [3.4, 3.1], rome: [3.5, 2.1], spain: [2.2, 3.4], madrid: [2.3, 2.45]
  };
  var names = Object.keys(base);
  var opts = names.map(function (n) { return { value: n, label: n }; });
  var ctrl = DL.controls(root);
  var A = DL.select(ctrl, { label: 'a', value: 'man', options: opts }, draw);
  var B = DL.select(ctrl, { label: 'is to b', value: 'king', options: opts }, draw);
  var Cc = DL.select(ctrl, { label: 'as c', value: 'woman', options: opts }, draw);
  var ctrl2 = DL.controls(root);
  var noise = DL.slider(ctrl2, { label: 'Noise on every vector', min: 0, max: 1.2, step: 0.05, value: 0 }, draw);
  var metric = DL.select(ctrl2, { label: 'Similarity', value: 'cos', options: [{ value: 'cos', label: 'cosine' }, { value: 'euc', label: 'Euclidean' }] }, draw);
  var excl = DL.toggle(ctrl2, 'Exclude a, b, c from the answer', true, draw);
  var c = DL.canvas(root, { aspect: 0.62 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Question: a is to b as c is to ?  Answer = nearest word to  v(b) - v(a) + v(c). Try man/king/woman (queen), then france/paris/germany (berlin). Turn off the exclusion toggle to see that the nearest word to the query is often one of the inputs (the famous "analogy" trick hides this). Add noise to see how fragile a parallelogram is. These 2D vectors are hand-built for illustration.' }));
  var P = DL.plot(c, { xmin: -2, xmax: 5, ymin: -1.5, ymax: 4.2, xlabel: 'dimension 1', ylabel: 'dimension 2' });
  function vecs() {
    var rand = DL.rng(5), s = noise.get(), v = {};
    names.forEach(function (n) { v[n] = [base[n][0] + s * 0.5 * DL.randn(rand), base[n][1] + s * 0.5 * DL.randn(rand)]; });
    return v;
  }
  function cosine(p, q) { return (p[0] * q[0] + p[1] * q[1]) / ((Math.hypot(p[0], p[1]) * Math.hypot(q[0], q[1])) || 1); }
  function arrow(ctx, x1, y1, x2, y2) {
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    var a = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - 8 * Math.cos(a - 0.4), y2 - 8 * Math.sin(a - 0.4)); ctx.lineTo(x2 - 8 * Math.cos(a + 0.4), y2 - 8 * Math.sin(a + 0.4)); ctx.closePath(); ctx.fill();
  }
  function draw() {
    var col = DL.colors(), ctx = c.ctx, v = vecs();
    var a = A.get(), b = B.get(), cc = Cc.get();
    var q = [v[b][0] - v[a][0] + v[cc][0], v[b][1] - v[a][1] + v[cc][1]];
    var scored = names.filter(function (n) { return !excl.get() || (n !== a && n !== b && n !== cc); }).map(function (n) {
      var s = metric.get() === 'cos' ? cosine(v[n], q) : -Math.hypot(v[n][0] - q[0], v[n][1] - q[1]);
      return { n: n, s: s };
    }).sort(function (x, y) { return y.s - x.s; });
    P.clear(); P.grid(); P.axes();
    ctx.save();
    ctx.font = '12px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    names.forEach(function (n) {
      var x = P.px(v[n][0]), y = P.py(v[n][1]);
      var isIn = (n === a || n === b || n === cc), isAns = scored.length && n === scored[0].n;
      ctx.fillStyle = isAns ? col.green : isIn ? col.accent : col.muted;
      ctx.beginPath(); ctx.arc(x, y, isIn || isAns ? 5.5 : 4, 0, 6.2832); ctx.fill();
      ctx.fillStyle = isAns ? col.green : isIn ? col.accent : col.ink;
      ctx.fillText(n, x + 8, y - 6);
    });
    // offset arrows a->b (blue) and c->query (accent dashed)
    ctx.strokeStyle = col.blue; ctx.fillStyle = col.blue; ctx.lineWidth = 2;
    arrow(ctx, P.px(v[a][0]), P.py(v[a][1]), P.px(v[b][0]), P.py(v[b][1]));
    ctx.strokeStyle = col.purple; ctx.fillStyle = col.purple; ctx.setLineDash([5, 4]);
    arrow(ctx, P.px(v[cc][0]), P.py(v[cc][1]), P.px(q[0]), P.py(q[1]));
    ctx.setLineDash([]);
    ctx.strokeStyle = col.amber; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(P.px(q[0]), P.py(q[1]), 6, 0, 6.2832); ctx.stroke();
    ctx.restore();
    var top = scored.slice(0, 3).map(function (s) { return '<b>' + s.n + '</b> (' + DL.fmt(s.s, 2) + ')'; }).join(', ');
    say('query = v(' + b + ') - v(' + a + ') + v(' + cc + ') = (' + DL.fmt(q[0], 2) + ', ' + DL.fmt(q[1], 2) + ')<br>nearest: ' + top);
  }
  c.draw = draw;
  draw();
});
