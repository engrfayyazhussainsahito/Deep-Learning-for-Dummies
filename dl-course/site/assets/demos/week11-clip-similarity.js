DL.register('week11-clip-similarity', function (root) {
  var NAMES = ['cat', 'dog', 'car', 'truck', 'plane', 'boat', 'apple', 'pizza'];
  var N = NAMES.length, D = 6, seed = 5;
  var ctrl = DL.controls(root), row = DL.row(root);
  var noiseS = DL.slider(ctrl, { label: 'Encoder noise σ', min: 0, max: 1.5, step: 0.01, value: 0.5 }, rebuild);
  var gapS = DL.slider(ctrl, { label: 'Modality gap', min: 0, max: 2, step: 0.01, value: 0.0 }, draw);
  var tauS = DL.slider(ctrl, { label: 'Temperature τ', min: -2, max: 0, step: 0.01, value: -1.3, format: function (v) { return DL.fmt(Math.pow(10, v), 3); } }, draw);
  var viewSel = DL.select(ctrl, { label: 'Matrix shows', value: 'row', options: [{ value: 'cos', label: 'cosine similarity' }, { value: 'row', label: 'softmax over texts (image to text)' }, { value: 'col', label: 'softmax over images (text to image)' }] }, draw);
  DL.button(row, 'New random encoders', function () { seed++; rebuild(); });
  var c = DL.canvas(root, { aspect: 0.62 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Illustrative synthetic embeddings (not real CLIP): each concept has a fixed 6-D direction, and image and text embeddings are that direction plus encoder noise. Rows are images, columns are captions. A strong diagonal means correct matches. Raise the noise to see retrieval errors. The modality gap is a constant offset between the image and text clouds (real CLIP models show one): cosine values drop but retrieval barely changes, while the temperature controls how peaked the softmax is.' }));
  var img = null, txt = null;
  function nrm(v) { var s = Math.sqrt(v.reduce(function (a, x) { return a + x * x; }, 0)) + 1e-12; return v.map(function (x) { return x / s; }); }
  function rebuild() {
    var rr = DL.rng(seed * 101), i, k, concept = [];
    function g() { return DL.randn(rr); }
    for (i = 0; i < N; i++) { var v = []; for (k = 0; k < D; k++) v.push(g()); concept.push(nrm(v)); }
    var s = noiseS.get(); img = []; txt = [];
    for (i = 0; i < N; i++) {
      img.push(concept[i].map(function (x) { return x + s * 0.45 * g(); }));
      txt.push(concept[i].map(function (x) { return x + s * 0.25 * g(); }));
    }
    draw();
  }
  function embeds() {
    var g = gapS.get(), off = []; for (var k = 0; k < D; k++) off.push(k === 0 ? g : (k === 1 ? -g * 0.5 : 0));
    return { I: img.map(function (v) { return nrm(v.map(function (x, k) { return x + off[k]; })); }),
             T: txt.map(function (v) { return nrm(v.map(function (x, k) { return x - off[k]; })); }) };
  }
  function draw() {
    if (!c.w || !img) return;
    var ctx = c.ctx, col = DL.colors(), e = embeds(), tau = Math.pow(10, tauS.get()), i, j, k;
    var S = [], R = [], C = [];
    for (i = 0; i < N; i++) { S.push([]); for (j = 0; j < N; j++) { var s = 0; for (k = 0; k < D; k++) s += e.I[i][k] * e.T[j][k]; S[i].push(s); } }
    for (i = 0; i < N; i++) R.push(DL.softmax(S[i].map(function (v) { return v / tau; })));
    for (j = 0; j < N; j++) C.push(DL.softmax(S.map(function (r) { return r[j] / tau; })));
    var view = viewSel.get(), M = [];
    for (i = 0; i < N; i++) { M.push([]); for (j = 0; j < N; j++) M[i].push(view === 'cos' ? S[i][j] : (view === 'row' ? R[i][j] : C[j][i])); }
    ctx.clearRect(0, 0, c.w, c.h);
    var left = 80, top = 40, sz = Math.max(80, Math.min(c.w - left - 10, c.h - top - 8));
    DL.heatmap(ctx, M, left, top, sz, sz, { min: view === 'cos' ? -1 : 0, max: 1, cmap: view === 'cos' ? DL.cmap.diverge : DL.cmap.viridis, labels: true, fmt: function (v) { return v.toFixed(2); }, rowLabels: NAMES.map(function (n) { return 'img: ' + n; }), colLabels: NAMES });
    ctx.fillStyle = col.ink; ctx.font = '12px ' + getComputedStyle(document.body).getPropertyValue('--font-sans'); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('columns: captions "a photo of a ..."', left, 4);
    var x2 = left + sz + 20;
    if (c.w - x2 > 130) {
      ctx.fillStyle = col.ink; ctx.fillText('image to text prediction', x2, 4);
      for (i = 0; i < N; i++) {
        var am = 0; for (j = 1; j < N; j++) if (S[i][j] > S[i][am]) am = j;
        ctx.fillStyle = am === i ? col.green : col.red; ctx.fillText((am === i ? 'ok  ' : 'err ') + NAMES[i] + ' → ' + NAMES[am], x2, top + (i + 0.3) * sz / N);
      }
    }
    var a1 = 0, a2 = 0, L1 = 0, L2 = 0;
    for (i = 0; i < N; i++) {
      var m1 = 0, m2 = 0; for (j = 1; j < N; j++) { if (S[i][j] > S[i][m1]) m1 = j; if (S[j][i] > S[m2][i]) m2 = j; }
      a1 += (m1 === i) ? 1 : 0; a2 += (m2 === i) ? 1 : 0; L1 -= Math.log(R[i][i] + 1e-300); L2 -= Math.log(C[i][i] + 1e-300);
    }
    say('image→text top-1: <b>' + a1 + '/' + N + '</b> &nbsp; text→image top-1: <b>' + a2 + '/' + N + '</b> &nbsp; CLIP loss = ½(' + DL.fmt(L1 / N, 3) + ' + ' + DL.fmt(L2 / N, 3) + ') = <b>' + DL.fmt((L1 + L2) / (2 * N), 3) + '</b> &nbsp; τ = ' + DL.fmt(tau, 3) + ' (logit scale ' + DL.fmt(1 / tau, 1) + ')');
  }
  c.draw = draw;
  rebuild();
});
