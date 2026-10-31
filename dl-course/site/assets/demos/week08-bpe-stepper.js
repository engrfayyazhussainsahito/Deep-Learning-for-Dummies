DL.register('week08-bpe-stepper', function (root) {
  var PRESETS = {
    lowest: 'low low low low low lower lower newest newest newest newest newest newest widest widest widest',
    morph: 'walk walks walked walking talk talks talked talking walk walk talk',
    custom: 'the cat the cat the cat sat the hat that cat'
  };
  var ctrl = DL.controls(root);
  var pre = DL.select(ctrl, { label: 'Corpus', value: 'lowest', options: [{ value: 'lowest', label: 'low / newest / widest' }, { value: 'morph', label: 'walk / talk + endings' }, { value: 'custom', label: 'the cat sat' }] }, function () { reset(); });
  var row = DL.row(root);
  var vocab, merges, view = DL.el('div', {});
  var tbl = DL.el('div', {});
  function reset() {
    var counts = {};
    PRESETS[pre.get()].split(' ').forEach(function (w) { counts[w] = (counts[w] || 0) + 1; });
    vocab = Object.keys(counts).sort().map(function (w) { return { s: w.split('').concat(['</w>']), n: counts[w], w: w }; });
    merges = []; draw();
  }
  function pairs() {
    var p = {};
    vocab.forEach(function (e) { for (var i = 0; i < e.s.length - 1; i++) { var k = e.s[i] + '\u0001' + e.s[i + 1]; p[k] = (p[k] || 0) + e.n; } });
    return p;
  }
  function best() {
    var p = pairs(), bk = null, bv = 0;
    Object.keys(p).sort().forEach(function (k) { if (p[k] > bv) { bv = p[k]; bk = k; } });
    return bk ? { a: bk.split('\u0001')[0], b: bk.split('\u0001')[1], n: bv } : null;
  }
  function step() {
    var b = best(); if (!b) return false;
    vocab.forEach(function (e) {
      var o = [], i = 0;
      while (i < e.s.length) {
        if (i < e.s.length - 1 && e.s[i] === b.a && e.s[i + 1] === b.b) { o.push(b.a + b.b); i += 2; } else { o.push(e.s[i]); i++; }
      }
      e.s = o;
    });
    merges.push(b); return true;
  }
  DL.button(row, 'Merge once', function () { step(); draw(); });
  DL.button(row, 'Merge 5', function () { for (var i = 0; i < 5; i++) step(); draw(); });
  DL.button(row, 'Reset', reset);
  root.appendChild(view);
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Each step counts all adjacent symbol pairs (weighted by word frequency) and fuses the most frequent pair into one new symbol; ties are broken alphabetically. "</w>" marks the end of a word so that "est</w>" at the end differs from "est" inside a word. Watch frequent endings (est</w>, er</w>, ing</w>) and stems become single tokens while the total token count falls.' }));
  function draw() {
    var h = '<table style="border-collapse:collapse;font-family:var(--font-mono);font-size:13px;width:100%"><tr><th style="text-align:left;padding:3px 8px">word</th><th style="text-align:right;padding:3px 8px">count</th><th style="text-align:left;padding:3px 8px">current segmentation</th></tr>';
    var total = 0, nw = 0;
    vocab.forEach(function (e) {
      total += e.s.length * e.n; nw += e.n;
      h += '<tr style="border-top:1px solid var(--c-line)"><td style="padding:3px 8px">' + e.w + '</td><td style="text-align:right;padding:3px 8px">' + e.n + '</td><td style="padding:3px 8px">' +
        e.s.map(function (t) { return '<span style="display:inline-block;margin:1px 2px;padding:1px 6px;border-radius:5px;border:1px solid ' + (t.length > 1 && t !== '</w>' ? 'var(--c-amber)' : 'var(--c-line)') + '">' + t.replace('<', '&lt;').replace('>', '&gt;') + '</span>'; }).join('') + '</td></tr>';
    });
    h += '</table>';
    view.innerHTML = h;
    var nb = best(), last = merges[merges.length - 1];
    var syms = {}; vocab.forEach(function (e) { e.s.forEach(function (t) { syms[t] = 1; }); });
    say('merges so far: <b>' + merges.length + '</b>' + (last ? ' (last: ' + last.a + ' + ' + last.b + ' → ' + last.a + last.b + ', count ' + last.n + ')' : '') +
      '<br>tokens in corpus: <b>' + total + '</b> for ' + nw + ' words (' + DL.fmt(total / nw, 2) + ' per word); symbols in use: ' + Object.keys(syms).length +
      '<br>next merge: ' + (nb ? '<b>' + (nb.a + ' + ' + nb.b).replace(/</g, '&lt;') + '</b> (count ' + nb.n + ')' : 'none (every word is one token)'));
  }
  reset();
});
