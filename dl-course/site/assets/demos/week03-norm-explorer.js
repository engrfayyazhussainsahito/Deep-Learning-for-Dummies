DL.register('week03-norm-explorer', function (root) {
  var ctrl = DL.controls(root);
  var mode = DL.select(ctrl, { label: 'Normalization', value: 'bn', options: [{ value: 'bn', label: 'BatchNorm (over N, H, W)' }, { value: 'ln', label: 'LayerNorm (over C, H, W)' }, { value: 'in', label: 'InstanceNorm (over H, W)' }, { value: 'gn', label: 'GroupNorm (groups of channels)' }] }, draw);
  var groups = DL.slider(ctrl, { label: 'GroupNorm groups G', min: 1, max: 8, step: 1, value: 2 }, draw);
  var skew = DL.slider(ctrl, { label: 'Per-sample scale spread', min: 0, max: 3, step: 0.1, value: 1.2 }, function () { gen(); draw(); });
  var row = DL.row(root);
  var evalT = DL.toggle(row, 'eval mode (use running statistics)', false, draw);
  var sub = DL.slider(row, { label: 'Batch used for stats', min: 1, max: 8, step: 1, value: 8 }, draw);
  DL.button(row, 'Resample', function () { seed++; gen(); draw(); });
  var c = DL.canvas(root, { aspect: 0.62 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Each row is one of N = 8 samples; each column block is one of C = 8 channels, drawn as a strip of 16 spatial values. Left: raw activations (samples have different scales and channels different means). Right: after normalization. Rows that share statistics are outlined. Try BatchNorm with batch used for stats = 1 (statistics get very noisy), then switch on eval mode, then switch to GroupNorm: the result is independent of batch size.' }));
  var N = 8, C = 8, M = 16, seed = 3, X;
  function gen() {
    var r = DL.rng(seed * 101);
    X = [];
    var cm = [], cs = [];
    for (var k = 0; k < C; k++) { cm[k] = 2.5 * DL.randn(r); cs[k] = 0.6 + Math.abs(DL.randn(r)); }
    for (var n = 0; n < N; n++) {
      var sc = Math.exp(skew.get() * 0.5 * DL.randn(r));
      X[n] = [];
      for (var k2 = 0; k2 < C; k2++) { X[n][k2] = []; for (var m = 0; m < M; m++) X[n][k2][m] = sc * (cm[k2] + cs[k2] * DL.randn(r)); }
    }
  }
  var runMean = null, runVar = null;
  // statistic sets: list of arrays of [n,k] cell indices sharing stats
  function sets(md) {
    var S = [], n, k, g;
    var nb = md === 'bn' ? Math.min(sub.get(), N) : N;
    if (md === 'bn') { for (k = 0; k < C; k++) { var a = []; for (n = 0; n < nb; n++) a.push([n, k]); S.push(a); } }
    else if (md === 'ln') { for (n = 0; n < N; n++) { var b = []; for (k = 0; k < C; k++) b.push([n, k]); S.push(b); } }
    else if (md === 'in') { for (n = 0; n < N; n++) for (k = 0; k < C; k++) S.push([[n, k]]); }
    else { var G = Math.min(groups.get(), C); while (C % G !== 0) G--; var per = C / G; for (n = 0; n < N; n++) for (g = 0; g < G; g++) { var d = []; for (k = g * per; k < (g + 1) * per; k++) d.push([n, k]); S.push(d); } }
    return S;
  }
  function stats(set) { var s = 0, q = 0, cnt = 0; set.forEach(function (p) { X[p[0]][p[1]].forEach(function (v) { s += v; q += v * v; cnt++; }); }); var mu = s / cnt; return [mu, Math.max(0, q / cnt - mu * mu)]; }
  function draw() {
    if (!X) gen();
    var col = DL.colors(), ctx = c.ctx, md = mode.get();
    ctx.clearRect(0, 0, c.w, c.h);
    var SS = sets(md), Y = [], k, n;
    for (n = 0; n < N; n++) { Y[n] = []; for (k = 0; k < C; k++) Y[n][k] = new Array(M); }
    var mus = [], vars = [];
    SS.forEach(function (set) {
      var st = stats(set), mu = st[0], vr = st[1];
      if (md === 'bn' && evalT.get()) { // running stats from the FULL batch
        var all = []; for (var i = 0; i < N; i++) all.push([i, set[0][1]]);
        var st2 = stats(all); mu = st2[0]; vr = st2[1];
      }
      mus.push(mu); vars.push(vr);
      set.forEach(function (p) { for (var m = 0; m < M; m++) Y[p[0]][p[1]][m] = (X[p[0]][p[1]][m] - mu) / Math.sqrt(vr + 1e-5); });
    });
    if (md === 'bn') { // samples beyond the stats subset: normalised with the same channel stats (as in training they would be absent)
      for (n = Math.min(sub.get(), N); n < N; n++) for (k = 0; k < C; k++) {
        var set = SS[k], st = stats(set), mu = st[0], vr = st[1];
        for (var m = 0; m < M; m++) Y[n][k][m] = (X[n][k][m] - mu) / Math.sqrt(vr + 1e-5);
      }
    }
    // layout
    var padL = 34, padT = 26, half = (c.w - padL - 30) / 2, cellH = Math.min(26, (c.h - padT - 10) / (N * C / 4)), blockW = half;
    function panel(M2, x0, title, lo, hi) {
      ctx.fillStyle = col.ink; ctx.font = '12px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      ctx.fillText(title, x0, 16);
      // simpler rendering: per (n,k) draw a small strip of M cells
      var cwid = (blockW - 4) / C, strip = cwid - 3, rowH = (c.h - padT - 14) / N;
      for (var n3 = 0; n3 < N; n3++) for (var k3 = 0; k3 < C; k3++) for (var m3 = 0; m3 < M; m3++) {
        var u2 = (M2[n3][k3][m3] - lo) / (hi - lo);
        ctx.fillStyle = DL.cmap.diverge(Math.max(0, Math.min(1, u2)));
        ctx.fillRect(x0 + k3 * cwid + m3 * strip / M, padT + n3 * rowH + 1, strip / M + 0.6, rowH - 3);
      }
      return { cwid: cwid, rowH: rowH };
    }
    var all = []; X.forEach(function (a) { a.forEach(function (b) { b.forEach(function (v) { all.push(v); }); }); });
    var mx = Math.max.apply(null, all.map(Math.abs)) || 1;
    var g1 = panel(X, padL, 'raw activations (rows: samples, columns: channels)', -mx, mx);
    var g2 = panel(Y, padL + half + 14, 'after normalization (colour range ±3)', -3, 3);
    ctx.fillStyle = col.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.font = '11px sans-serif';
    for (n = 0; n < N; n++) ctx.fillText('n' + n, padL - 5, padT + (n + 0.5) * g1.rowH);
    // outline statistic sets on both panels
    ctx.lineWidth = 1.4; ctx.strokeStyle = col.ink; ctx.globalAlpha = 0.7;
    [padL, padL + half + 14].forEach(function (x0) {
      SS.forEach(function (set) {
        var ns = set.map(function (p) { return p[0]; }), ks = set.map(function (p) { return p[1]; });
        var n0 = Math.min.apply(null, ns), n1 = Math.max.apply(null, ns), k0 = Math.min.apply(null, ks), k1 = Math.max.apply(null, ks);
        ctx.strokeRect(x0 + k0 * g1.cwid - 1, padT + n0 * g1.rowH, (k1 - k0 + 1) * g1.cwid - 1, (n1 - n0 + 1) * g1.rowH - 1);
      });
    });
    ctx.globalAlpha = 1;
    var sdY = 0; var ySum = 0, yq = 0, yc = 0; Y.forEach(function (a) { a.forEach(function (b) { b.forEach(function (v) { ySum += v; yq += v * v; yc++; }); }); });
    var mu0 = ySum / yc; sdY = Math.sqrt(yq / yc - mu0 * mu0);
    var nStat = SS.length, mm = Math.max.apply(null, mus.map(Math.abs)), vv = vars.reduce(function (a, b) { return a + b; }, 0) / vars.length;
    var msg = md === 'bn' ? (evalT.get() ? 'eval: each channel uses fixed (running) statistics, so a sample is normalised independent of its batch mates' : 'train: statistics come from ' + Math.min(sub.get(), N) + ' sample(s) per channel; a batch of 1 gives a degenerate estimate') : 'statistics per sample: independent of the rest of the batch (train = eval)';
    say('<b>' + nStat + '</b> statistic groups, each of ' + SS[0].length + ' channel-slice(s) x ' + M + ' values &nbsp; output mean ' + DL.fmt(mu0, 2) + ', std ' + DL.fmt(sdY, 2) + '<br>' + msg);
  }
  c.draw = draw;
  gen(); 
  draw();
});
