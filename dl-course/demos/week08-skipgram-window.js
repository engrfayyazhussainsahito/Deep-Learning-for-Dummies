DL.register('week08-skipgram-window', function (root) {
  var SENT = {
    a: 'the quick brown fox jumps over the lazy dog'.split(' '),
    b: 'we train word vectors by predicting nearby words'.split(' '),
    c: 'a cat sat on the mat while the dog slept on the rug'.split(' ')
  };
  // Pretend corpus counts (illustrative) for the noise distribution
  var COUNT = { the: 900, a: 500, on: 300, we: 80, dog: 60, cat: 40, over: 45, by: 70, while: 30, mat: 12, rug: 8, fox: 9, quick: 10, brown: 14, lazy: 6, jumps: 5, sat: 20, slept: 7, train: 18, word: 35, vectors: 15, predicting: 4, nearby: 3, words: 40 };
  var ctrl = DL.controls(root);
  var sel = DL.select(ctrl, { label: 'Sentence', value: 'a', options: [{ value: 'a', label: 'quick brown fox' }, { value: 'b', label: 'word vectors' }, { value: 'c', label: 'cat and dog' }] }, function () { pos = 0; draw(); });
  var win = DL.slider(ctrl, { label: 'Window size m', min: 1, max: 4, step: 1, value: 2 }, draw);
  var K = DL.slider(ctrl, { label: 'Negatives K', min: 1, max: 8, step: 1, value: 3 }, resample);
  var alpha = DL.slider(ctrl, { label: 'Noise exponent α', min: 0, max: 1.5, step: 0.05, value: 0.75 }, resample);
  var row = DL.row(root);
  var pos = 0, negs = [], seedN = 1;
  DL.button(row, 'Next centre word', function () { pos = (pos + 1) % SENT[sel.get()].length; resample(); });
  DL.button(row, 'Resample negatives', function () { seedN++; resample(); });
  var out = DL.el('div', {}); root.appendChild(out);
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'The centre word (accent) generates one (centre, context) pair for every word inside the window. For each pair, K "noise" words are drawn from the vocabulary with probability ∝ count^α. With α = 1 you mostly draw "the" and "a"; α = 0.75 (the word2vec default) flattens the distribution; α = 0 is uniform. Counts here are illustrative.' }));
  function dist() {
    var ws = Object.keys(COUNT), p = ws.map(function (w) { return Math.pow(COUNT[w], alpha.get()); }), s = DL.sum(p);
    return { ws: ws, p: p.map(function (x) { return x / s; }) };
  }
  function resample() {
    var d = dist(), rand = DL.rng(seedN * 977 + pos * 31 + 7);
    negs = [];
    for (var k = 0; k < Math.round(K.get()); k++) {
      var u = rand(), acc = 0, j = 0;
      for (j = 0; j < d.p.length; j++) { acc += d.p[j]; if (u <= acc) break; }
      negs.push(d.ws[Math.min(j, d.ws.length - 1)]);
    }
    draw();
  }
  function chip(w, bg, fg, extra) {
    return '<span style="display:inline-block;margin:2px 3px;padding:3px 9px;border-radius:12px;border:1px solid var(--c-line);background:' + bg + ';color:' + fg + ';font-family:var(--font-mono);font-size:13px;' + (extra || '') + '">' + w + '</span>';
  }
  function draw() {
    var s = SENT[sel.get()], m = Math.round(win.get()), h = '<div style="margin:6px 0">';
    s.forEach(function (w, i) {
      var inWin = i !== pos && Math.abs(i - pos) <= m;
      if (i === pos) h += chip(w, 'var(--c-accent)', '#fff', 'font-weight:700');
      else if (inWin) h += chip(w, 'var(--c-soft)', 'var(--c-ink)', 'border-color:var(--c-blue);border-width:2px');
      else h += chip(w, 'transparent', 'var(--c-muted)');
    });
    h += '</div>';
    var ctx = [];
    s.forEach(function (w, i) { if (i !== pos && Math.abs(i - pos) <= m) ctx.push(w); });
    h += '<div style="margin:6px 0;font-size:13px"><b>Positive pairs</b> (centre, context): ' + ctx.map(function (w) { return chip('(' + s[pos] + ', ' + w + ')', 'transparent', 'var(--c-green)'); }).join('') + '</div>';
    h += '<div style="margin:6px 0;font-size:13px"><b>Negatives for the first pair</b> (' + s[pos] + ', ?): ' + negs.map(function (w) { return chip(w, 'transparent', 'var(--c-red)'); }).join('') + '</div>';
    out.innerHTML = h;
    var d = dist(), top = d.ws.map(function (w, i) { return { w: w, p: d.p[i] }; }).sort(function (a, b) { return b.p - a.p; }).slice(0, 3);
    say('window 2m = ' + (2 * m) + ' → up to ' + (2 * m) + ' positive pairs per centre word; each costs 1 + K = ' + (1 + Math.round(K.get())) + ' dot products.<br>Top noise probabilities: ' + top.map(function (t) { return t.w + ' ' + DL.fmt(t.p, 3); }).join(', '));
  }
  resample();
});
