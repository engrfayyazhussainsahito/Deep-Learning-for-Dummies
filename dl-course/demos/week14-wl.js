DL.register('week14-wl', function (root) {
  // Two graphs; run 1-WL colour refinement jointly so colours are comparable across graphs.
  function ring(k, off) { var e = []; for (var i = 0; i < k; i++) e.push([off + i, off + (i + 1) % k]); return e; }
  var PRESETS = {
    c6: { name: 'C6 vs two triangles (1-WL fails)', n1: 6, e1: ring(6, 0), n2: 6, e2: ring(3, 0).concat(ring(3, 3)) },
    path: { name: 'path P5 vs star-ish tree (1-WL succeeds)', n1: 5, e1: [[0, 1], [1, 2], [2, 3], [3, 4]], n2: 5, e2: [[0, 1], [0, 2], [0, 3], [3, 4]] },
    reg: { name: 'prism vs K3,3 (3-regular, 1-WL fails)', n1: 6, e1: ring(3, 0).concat(ring(3, 3), [[0, 3], [1, 4], [2, 5]]), n2: 6, e2: [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]] },
    cyc: { name: 'C5 + tail vs C4 + 2 tail (1-WL succeeds)', n1: 6, e1: ring(5, 0).concat([[0, 5]]), n2: 6, e2: ring(4, 0).concat([[0, 4], [4, 5]]) }
  };
  var G, cols1, cols2, round, hist;
  var ctrl = DL.controls(root);
  var sel = DL.select(ctrl, { label: 'Graph pair', value: 'c6', options: Object.keys(PRESETS).map(function (k) { return { value: k, label: PRESETS[k].name }; }) }, function () { load(); });
  var row = DL.row(root);
  DL.button(row, 'Refine one round', step);
  DL.button(row, 'Reset', function () { load(); });
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Each round, a node gets a new colour from the pair (its own colour, the multiset of its neighbours\' colours). Identical signatures get identical colours in both graphs. When the colour histograms differ, 1-WL has distinguished the graphs; if the partition stops changing with equal histograms, it never will.' }));

  function adj(n, E) { var a = []; for (var i = 0; i < n; i++) a.push([]); E.forEach(function (e) { a[e[0]].push(e[1]); a[e[1]].push(e[0]); }); return a; }
  function layout(n, E, cx, cy, r) {
    // circular layout; for the two-triangle graph use two small circles
    var p = [], i;
    if (G.key === 'c6' && cx > 0.5) { for (i = 0; i < n; i++) { var g = i < 3 ? 0 : 1, a0 = (i % 3) / 3 * 6.2832 - 1.57; p.push([cx + (g ? 0.1 : -0.1) + 0.075 * Math.cos(a0), cy + 0.09 * Math.sin(a0) * 1.0]); } return p; }
    for (i = 0; i < n; i++) { var a = i / n * 6.2832 - 1.57; p.push([cx + r * Math.cos(a), cy + r * Math.sin(a) * 1.25]); }
    return p;
  }
  function load() {
    var P = PRESETS[sel.get()];
    G = { key: sel.get(), n1: P.n1, n2: P.n2, a1: adj(P.n1, P.e1), a2: adj(P.n2, P.e2), E1: P.e1, E2: P.e2 };
    G.p1 = layout(P.n1, P.e1, 0.25, 0.46, 0.17); G.p2 = layout(P.n2, P.e2, 0.75, 0.46, 0.17);
    cols1 = []; cols2 = []; var i;
    for (i = 0; i < P.n1; i++) cols1.push(0);
    for (i = 0; i < P.n2; i++) cols2.push(0);
    round = 0; hist = [histOf(cols1, cols2)]; draw();
  }
  function histOf(a, b) {
    var ha = {}, hb = {};
    a.forEach(function (x) { ha[x] = (ha[x] || 0) + 1; });
    b.forEach(function (x) { hb[x] = (hb[x] || 0) + 1; });
    return { a: ha, b: hb };
  }
  function sameHist(h) { var k; for (k in h.a) if (h.a[k] !== h.b[k]) return false; for (k in h.b) if (h.a[k] !== h.b[k]) return false; return true; }
  function step() {
    var sig = {}, nextId = 0;
    function sigOf(colors, a, i) { return colors[i] + '|' + a[i].map(function (j) { return colors[j]; }).sort(function (x, y) { return x - y; }).join(','); }
    var s1 = cols1.map(function (_, i) { return sigOf(cols1, G.a1, i); });
    var s2 = cols2.map(function (_, i) { return sigOf(cols2, G.a2, i); });
    var all = s1.concat(s2).filter(function (s, i, arr) { return arr.indexOf(s) === i; }).sort();
    all.forEach(function (s) { sig[s] = nextId++; });
    var prevK = Object.keys(histOf(cols1, cols2).a).length;
    cols1 = s1.map(function (s) { return sig[s]; }); cols2 = s2.map(function (s) { return sig[s]; });
    round++; hist.push(histOf(cols1, cols2)); draw();
  }
  function nclasses() { var s = {}; cols1.concat(cols2).forEach(function (x) { s[x] = 1; }); return Object.keys(s).length; }
  var lastK = {};
  function draw() {
    if (!c.w || !G) return;
    var ctx = c.ctx, col = DL.colors(), W = c.w, H = c.h;
    ctx.clearRect(0, 0, W, H);
    var pal = [col.blue, col.amber, col.green, col.purple, col.red, col.teal, col.accent, col.muted, col.ink];
    function drawG(p, E, cs, title, x0) {
      ctx.strokeStyle = col.muted; ctx.lineWidth = 1.5;
      E.forEach(function (e) { ctx.beginPath(); ctx.moveTo(p[e[0]][0] * W, p[e[0]][1] * H); ctx.lineTo(p[e[1]][0] * W, p[e[1]][1] * H); ctx.stroke(); });
      p.forEach(function (q, i) { ctx.beginPath(); ctx.arc(q[0] * W, q[1] * H, 15, 0, 6.2832); ctx.fillStyle = pal[cs[i] % pal.length]; ctx.fill(); ctx.strokeStyle = col.ink; ctx.lineWidth = 1.2; ctx.stroke(); ctx.fillStyle = col.bg; ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(cs[i]), q[0] * W, q[1] * H); });
      ctx.fillStyle = col.ink; ctx.font = '13px sans-serif'; ctx.textAlign = 'center'; ctx.fillText(title, x0 * W, 18);
    }
    drawG(G.p1, G.E1, cols1, 'Graph A', 0.25); drawG(G.p2, G.E2, cols2, 'Graph B', 0.75);
    // histograms
    var h = hist[hist.length - 1], keys = {}, k;
    for (k in h.a) keys[k] = 1; for (k in h.b) keys[k] = 1;
    var ks = Object.keys(keys).sort(function (x, y) { return x - y; });
    var y0 = H * 0.80, bw = Math.min(34, (W - 40) / (ks.length * 2.4 + 1));
    ctx.font = '11px sans-serif'; ctx.textAlign = 'center'; ctx.fillStyle = col.muted;
    ctx.fillText('colour histogram: A (left bar) vs B (right bar)', W / 2, y0 - 22);
    ks.forEach(function (key, i) {
      var x = W / 2 - ks.length * bw * 1.2 + i * bw * 2.4 + bw * 0.6;
      var ha = h.a[key] || 0, hb = h.b[key] || 0;
      ctx.fillStyle = pal[key % pal.length]; ctx.fillRect(x, y0 + 30 - ha * 6, bw * 0.45, ha * 6); ctx.fillRect(x + bw * 0.55, y0 + 30 - hb * 6, bw * 0.45, hb * 6);
      ctx.fillStyle = col.muted; ctx.fillText('c' + key + ': ' + ha + '/' + hb, x + bw * 0.5, y0 + 44);
    });
    var diff = !sameHist(h);
    var kNow = nclasses();
    var stable = round > 0 && kNow === lastK[round - 1 + G.key];
    lastK[round + G.key] = kNow;
    say('round <b>' + round + '</b> &nbsp; colour classes: <b>' + kNow + '</b> &nbsp; histograms: ' + (diff ? '<b>DIFFERENT, graphs distinguished</b>' : '<b>identical</b>' + (stable ? ' and partition stable: 1-WL cannot separate these graphs' : '')));
  }
  c.draw = draw;
  load();
});
