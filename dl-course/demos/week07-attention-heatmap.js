DL.register('week07-attention-heatmap', function (root) {
  var EX = {
    reorder: { src: ['the', 'black', 'cat', 'is', 'sleeping', '<eos>'], tgt: ['le', 'chat', 'noir', 'dort', '<eos>'],
      align: [[0], [2], [1], [3, 4], [5]], note: 'English puts the adjective before the noun and French usually after it, so the alignment crosses.' },
    mono: { src: ['I', 'love', 'deep', 'learning', '<eos>'], tgt: ['yo', 'amo', 'el', 'aprendizaje', 'profundo', '<eos>'],
      align: [[0], [1], [3], [3], [2], [4]], note: 'Mostly monotone, with one local swap (deep learning becomes aprendizaje profundo).' },
    long: { src: ['she', 'has', 'never', 'seen', 'the', 'sea', '<eos>'], tgt: ['sie', 'hat', 'das', 'Meer', 'noch', 'nie', 'gesehen', '<eos>'],
      align: [[0], [1], [4], [5], [2], [2], [3], [6]], note: 'German moves the participle to the end and puts adverbs after the object: long-range reordering.' }
  };
  var ctrl = DL.controls(root);
  var ex = DL.select(ctrl, { label: 'Sentence pair', value: 'reorder', options: [{ value: 'reorder', label: 'EN to FR (adjective swap)' }, { value: 'mono', label: 'EN to ES (near-monotone)' }, { value: 'long', label: 'EN to DE (long-range)' }] }, function () { sel = -1; draw(); });
  var temp = DL.slider(ctrl, { label: 'Score sharpness 1/T', min: 0.2, max: 8, step: 0.1, value: 3 }, draw);
  var noise = DL.slider(ctrl, { label: 'Score noise (poorly trained model)', min: 0, max: 3, step: 0.1, value: 0.3 }, draw);
  var sel = -1;
  var c = DL.canvas(root, { aspect: 0.62 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Each row is one decoder step; each column a source word; a cell is the attention weight a_ti (rows sum to 1). Scores here are synthetic: high for the aligned source word, low elsewhere, plus noise, then softmax with sharpness 1/T. Click a row to see its entropy and context vector composition. Low sharpness gives diffuse weights (near-average pooling); high sharpness approaches hard alignment.' }));
  function build() {
    var e = EX[ex.get()], rand = DL.rng(5), W = [], i, j;
    for (i = 0; i < e.tgt.length; i++) {
      var sc = [];
      for (j = 0; j < e.src.length; j++) {
        var s = 0; e.align[i].forEach(function (a) { s = Math.max(s, a === j ? 1 : (Math.abs(a - j) === 1 ? 0.25 : 0)); });
        sc.push(s * 2 + noise.get() * DL.randn(rand));
      }
      W.push(DL.softmax(sc.map(function (v) { return v * temp.get(); })));
    }
    return W;
  }
  function draw() {
    var e = EX[ex.get()], W = build(), ctx = c.ctx, col = DL.colors();
    ctx.clearRect(0, 0, c.w, c.h);
    var x0 = 78, y0 = 52, gw = Math.min(c.w - x0 - 20, 520), gh = Math.min(c.h - y0 - 14, 300);
    DL.heatmap(ctx, W, x0, y0, gw, gh, { min: 0, max: 1, cmap: DL.cmap.viridis, labels: true, rowLabels: e.tgt, colLabels: e.src, fmt: function (v) { return v.toFixed(2); } });
    ctx.fillStyle = col.muted; ctx.font = '11px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('source →', x0, 14); ctx.fillText('target ↓', 6, y0 - 14);
    if (sel >= 0 && sel < W.length) { ctx.strokeStyle = col.accent; ctx.lineWidth = 2.5; ctx.strokeRect(x0, y0 + sel * gh / W.length, gw, gh / W.length); }
    c._geo = { x0: x0, y0: y0, gw: gw, gh: gh, n: W.length };
    var ent = W.map(function (r) { var s = 0; r.forEach(function (p) { s -= p > 0 ? p * Math.log(p) : 0; }); return s; });
    var txt = '<i>' + e.note + '</i><br>mean entropy = <b>' + DL.fmt(DL.sum(ent) / ent.length, 2) + '</b> nats (uniform would be ' + DL.fmt(Math.log(e.src.length), 2) + ')';
    if (sel >= 0 && sel < W.length) {
      var top = W[sel].map(function (p, j) { return [p, j]; }).sort(function (a, b) { return b[0] - a[0]; }).slice(0, 3);
      txt += '<br>Row "' + e.tgt[sel] + '": context ≈ ' + top.map(function (t) { return DL.fmt(t[0], 2) + '·h(' + e.src[t[1]] + ')'; }).join(' + ') + ' + ...; entropy ' + DL.fmt(ent[sel], 2);
    } else txt += '<br>Click a row for its context-vector composition.';
    say(txt);
  }
  c.canvas.addEventListener && c.canvas.addEventListener('click', function (ev) {
    var r = c.canvas.getBoundingClientRect(), y = ev.clientY - r.top, g = c._geo;
    if (!g) return;
    var k = Math.floor((y - g.y0) / (g.gh / g.n));
    sel = k >= 0 && k < g.n ? k : -1; draw();
  });
  c.draw = draw;
  draw();
});
