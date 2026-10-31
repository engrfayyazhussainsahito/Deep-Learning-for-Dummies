DL.register('week07-beam-search', function (root) {
  // A hand-built bigram "language model": P(next | previous word). Rows sum to 1.
  var LM = {
    '<s>': { 'the': 0.55, 'a': 0.45 },
    'the': { 'cat': 0.34, 'dog': 0.33, 'mat': 0.33 },
    'a': { 'dog': 0.9, 'cat': 0.05, 'mat': 0.05 },
    'cat': { '</s>': 0.4, 'sat': 0.3, 'ran': 0.3 },
    'dog': { 'ran': 0.8, 'sat': 0.1, '</s>': 0.1 },
    'mat': { '</s>': 0.6, 'sat': 0.2, 'ran': 0.2 },
    'sat': { 'on': 0.7, '</s>': 0.3 },
    'ran': { 'on': 0.3, '</s>': 0.7 },
    'on': { 'the': 0.5, 'a': 0.5 }
  };
  var MAXLEN = 8;
  var ctrl = DL.controls(root);
  var B = DL.slider(ctrl, { label: 'Beam width B', min: 1, max: 6, step: 1, value: 2 }, rebuild);
  var alpha = DL.slider(ctrl, { label: 'Length penalty α (score / len^α)', min: 0, max: 1.5, step: 0.1, value: 0 }, rebuild);
  var row = DL.row(root);
  var steps = [], shown = 0, fin = [];
  function norm(h) { return h.score / Math.pow(Math.max(1, h.words.length), alpha.get()); }
  function rebuild() {
    var b = Math.round(B.get()), beams = [{ words: [], score: 0, last: '<s>' }];
    steps = []; fin = [];
    for (var t = 0; t < MAXLEN && beams.length; t++) {
      var cand = [];
      beams.forEach(function (h) {
        var nxt = LM[h.last] || {};
        Object.keys(nxt).forEach(function (w) { cand.push({ words: h.words.concat([w]), score: h.score + Math.log(nxt[w]), last: w, parent: h }); });
      });
      cand.sort(function (a, b2) { return norm(b2) - norm(a) || b2.score - a.score; });
      var kept = [], pruned = [];
      cand.forEach(function (h) { if (kept.length < b) kept.push(h); else pruned.push(h); });
      steps.push({ kept: kept, pruned: pruned });
      beams = [];
      kept.forEach(function (h) { if (h.last === '</s>') fin.push(h); else beams.push(h); });
    }
    fin.sort(function (a, b2) { return norm(b2) - norm(a); });
    shown = Math.min(shown, steps.length);
    render();
  }
  function html(h, cls) {
    return '<div style="margin:1px 0;' + (cls === 'dead' ? 'opacity:.45;text-decoration:line-through;' : '') + '">' + h.words.join(' ') + ' &nbsp;<span style="color:var(--c-muted)">log p = ' + DL.fmt(h.score, 2) + ', p = ' + DL.fmt(Math.exp(h.score), 3) + '</span></div>';
  }
  var out = DL.el('div', {});
  DL.button(row, 'Next step', function () { if (shown < steps.length) shown++; render(); });
  DL.button(row, 'Run to end', function () { shown = steps.length; render(); });
  DL.button(row, 'Reset', function () { shown = 0; render(); });
  root.appendChild(out);
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'The model is a tiny hand-made bigram LM (the next word depends only on the previous one). Greedy search (B = 1) grabs "the" because it is the most likely first word, but every continuation after "the" is flat, so the finished sentence is weak. B = 2 keeps "a" alive and finds the globally better "a dog ran". Raise α to favour longer hypotheses and watch the ranking change.' }));
  function render() {
    var s = '';
    for (var t = 0; t < shown; t++) {
      s += '<div style="margin:8px 0 2px;font-weight:600">Step ' + (t + 1) + ': expand then keep the best ' + Math.round(B.get()) + '</div>';
      steps[t].kept.forEach(function (h) { s += html(h, 'kept'); });
      steps[t].pruned.slice(0, 4).forEach(function (h) { s += html(h, 'dead'); });
      if (steps[t].pruned.length > 4) s += '<div style="opacity:.45">... ' + (steps[t].pruned.length - 4) + ' more pruned</div>';
    }
    if (!shown) s = '<div>Press "Next step" to expand the search one token at a time.</div>';
    out.innerHTML = '<div style="font-family:var(--font-mono);font-size:12.5px;max-height:300px;overflow:auto;padding:8px 10px;border:1px solid var(--c-line);border-radius:6px">' + s + '</div>';
    var done = shown === steps.length;
    var finTxt = fin.length ? fin.slice(0, 3).map(function (h, k) { return (k + 1) + '. <b>' + h.words.slice(0, -1).join(' ') + '</b> (p = ' + DL.fmt(Math.exp(h.score), 3) + ', normalized score ' + DL.fmt(norm(h), 3) + ')'; }).join('<br>') : 'no finished hypothesis yet';
    say(done ? 'Finished hypotheses, best first:<br>' + finTxt : 'Step ' + shown + ' of ' + steps.length + '. Press "Run to end" to see the ranking.');
  }
  rebuild();
});
