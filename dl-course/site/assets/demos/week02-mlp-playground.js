DL.register('week02-mlp-playground', function (root) {
  var DATA = { circles: 'Concentric circles', xor: 'XOR blobs', spiral: 'Two spirals', moons: 'Two moons' };
  var ACTS = {
    tanh: { f: Math.tanh, d: function (z, a) { return 1 - a * a; } },
    relu: { f: function (z) { return z > 0 ? z : 0; }, d: function (z) { return z > 0 ? 1 : 0; } },
    sigmoid: { f: DL.sigmoid, d: function (z, a) { return a * (1 - a); } },
    silu: { f: function (z) { return z * DL.sigmoid(z); }, d: function (z) { var s = DL.sigmoid(z); return s * (1 + z * (1 - s)); } }
  };
  var ds = 'circles', X = [], Y = [], Hn = 8, Ln = 2, act = 'tanh', net = null, epoch = 0, losses = [], acc = 0, running = false, raf = 0, visible = true, last = 0;
  var rand = DL.rng(5);
  var ctrl = DL.controls(root);
  DL.select(ctrl, { label: 'Dataset', value: ds, options: Object.keys(DATA).map(function (k) { return { value: k, label: DATA[k] }; }) }, function (v) { ds = v; makeData(); init(); });
  DL.select(ctrl, { label: 'Activation', value: act, options: Object.keys(ACTS).map(function (k) { return { value: k, label: k }; }) }, function (v) { act = v; init(); });
  DL.slider(ctrl, { label: 'Hidden layers', min: 1, max: 4, step: 1, value: Ln, format: function (v) { return String(Math.round(v)); } }, function (v) { Ln = Math.round(v); init(); });
  DL.slider(ctrl, { label: 'Units per layer', min: 2, max: 24, step: 1, value: Hn, format: function (v) { return String(Math.round(v)); } }, function (v) { Hn = Math.round(v); init(); });
  var lrS = DL.slider(ctrl, { label: 'Learning rate', min: 0.005, max: 0.5, step: 0.005, value: 0.1, format: function (v) { return (+v).toFixed(3); } }, function () {});
  var row = DL.row(root);
  var runBtn = DL.button(row, 'Train', function () { setRun(!running); });
  DL.button(row, 'Re-initialise', function () { init(); });
  var c = DL.canvas(root, { aspect: 0.5 });
  var P = DL.plot(c, { xmin: -1.5, xmax: 1.5, ymin: -1.5, ymax: 1.5, xlabel: 'x1', ylabel: 'x2', pad: { l: 40, r: 190, t: 10, b: 32 } });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'A real MLP (softmax + cross-entropy, full-batch gradient descent, backprop written out in this file) trains in your browser. Background = predicted probability of class 1. Try: XOR with 1 layer of 2 units, spirals with 1 layer versus 3 layers, and sigmoid versus ReLU at 4 layers. The loss curve is on the right.' }));

  function makeData() {
    X = []; Y = []; var i, n = 120, t, r;
    for (i = 0; i < n; i++) {
      var k = i % 2;
      if (ds === 'circles') { t = rand() * 6.283; r = (k ? 0.45 : 1.0) + 0.07 * DL.randn(rand); X.push([r * Math.cos(t), r * Math.sin(t)]); }
      else if (ds === 'xor') { var sx = rand() < 0.5 ? -1 : 1, sy = k ? sx : -sx; X.push([0.6 * sx + 0.18 * DL.randn(rand), 0.6 * sy + 0.18 * DL.randn(rand)]); }
      else if (ds === 'spiral') { t = 0.3 + 3.2 * Math.sqrt(rand()); var a = t * 1.9 + (k ? Math.PI : 0); r = t / 3.6; X.push([r * Math.cos(a) + 0.03 * DL.randn(rand), r * Math.sin(a) + 0.03 * DL.randn(rand)]); }
      else { t = rand() * Math.PI; if (k) X.push([0.5 - Math.cos(t) * 0.9 + 0.5 * 0, 0.25 - Math.sin(t) * 0.7 + 0.07 * DL.randn(rand)]); else X.push([Math.cos(t) * 0.9 - 0.5, Math.sin(t) * 0.7 - 0.2 + 0.07 * DL.randn(rand)]); }
      Y.push(k);
    }
  }
  function init() {
    setRun(false);
    var sizes = [2], i, j, k; for (i = 0; i < Ln; i++) sizes.push(Hn); sizes.push(2);
    net = { W: [], b: [] };
    for (i = 0; i < sizes.length - 1; i++) {
      var sc = Math.sqrt((act === 'relu' ? 2 : 1) / sizes[i]), W = [];
      for (j = 0; j < sizes[i]; j++) { W.push([]); for (k = 0; k < sizes[i + 1]; k++) W[j].push(sc * DL.randn(rand)); }
      net.W.push(W); net.b.push(new Array(sizes[i + 1]).fill(0));
    }
    epoch = 0; losses = []; evalAll(); c.redraw();
  }
  function forward(x, keep) {
    var A = [x], Z = [], a = x, L = net.W.length, i, k, j;
    for (i = 0; i < L; i++) {
      var W = net.W[i], z = [];
      for (k = 0; k < W[0].length; k++) { var s = net.b[i][k]; for (j = 0; j < a.length; j++) s += a[j] * W[j][k]; z.push(s); }
      Z.push(z);
      a = i === L - 1 ? z : z.map(ACTS[act].f);
      A.push(a);
    }
    var p = DL.softmax(a);
    return keep ? { A: A, Z: Z, p: p } : p;
  }
  function evalAll() {
    var l = 0, ok = 0, i;
    for (i = 0; i < X.length; i++) { var p = forward(X[i]); l -= Math.log(Math.max(p[Y[i]], 1e-12)); if ((p[1] > 0.5 ? 1 : 0) === Y[i]) ok++; }
    acc = ok / X.length; var v = l / X.length; if (!isFinite(v)) v = 20; losses.push(v);
  }
  function stepOnce() {
    var L = net.W.length, n = X.length, gW = [], gb = [], i, j, k, s;
    for (i = 0; i < L; i++) { gW.push(net.W[i].map(function (r) { return r.map(function () { return 0; }); })); gb.push(net.b[i].map(function () { return 0; })); }
    for (s = 0; s < n; s++) {
      var f = forward(X[s], true), d = f.p.slice(); d[Y[s]] -= 1;           // dL/dz_L = p - onehot
      for (i = L - 1; i >= 0; i--) {
        var Ain = f.A[i], W = net.W[i], dprev = new Array(Ain.length).fill(0);
        for (k = 0; k < d.length; k++) {
          gb[i][k] += d[k] / n;
          for (j = 0; j < Ain.length; j++) { gW[i][j][k] += Ain[j] * d[k] / n; dprev[j] += W[j][k] * d[k]; }
        }
        if (i > 0) { var Zp = f.Z[i - 1], Ap = f.A[i]; d = dprev.map(function (v, q) { return v * ACTS[act].d(Zp[q], Ap[q]); }); }
      }
    }
    var lr = lrS.get();
    for (i = 0; i < L; i++) for (k = 0; k < net.b[i].length; k++) { net.b[i][k] -= lr * gb[i][k]; for (j = 0; j < net.W[i].length; j++) net.W[i][j][k] -= lr * gW[i][j][k]; }
    epoch++; evalAll();
  }
  function setRun(r) { running = r; runBtn.textContent = r ? 'Pause' : 'Train'; cancelAnimationFrame(raf); if (r && visible) { last = 0; raf = requestAnimationFrame(loop); } }
  function loop(t) {
    if (!running) return;
    if (!last || t - last > 25) { last = t; for (var i = 0; i < 4; i++) stepOnce(); c.redraw(); }
    if (epoch > 40000) { setRun(false); return; }
    raf = requestAnimationFrame(loop);
  }
  if (window.IntersectionObserver) new IntersectionObserver(function (es) { visible = es[0].isIntersecting; if (!visible) cancelAnimationFrame(raf); else if (running) { last = 0; raf = requestAnimationFrame(loop); } }).observe(root);

  function draw() {
    if (!c.w || !net) return;
    var col = DL.colors(), ctx = c.ctx, i;
    P.clear();
    P.heat(function (x, y) { return forward([x, y])[1]; }, DL.cmap.diverge, 8);
    P.axes();
    var x0 = [], y0 = [], x1 = [], y1 = [];
    for (i = 0; i < X.length; i++) { (Y[i] ? x1 : x0).push(X[i][0]); (Y[i] ? y1 : y0).push(X[i][1]); }
    ctx.save(); P.points(x0, y0, col.blue, 3.6); P.points(x1, y1, col.red, 3.6); ctx.restore();
    // loss curve panel
    var lx = c.w - 175, ly = 30, lw = 160, lh = c.h - 90, mx = 0.1, k;
    for (k = 0; k < losses.length; k++) mx = Math.max(mx, losses[k]);
    mx = Math.min(mx, 5);
    ctx.fillStyle = col.muted; ctx.font = '12px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillText('training loss', lx, ly - 8);
    ctx.strokeStyle = col.line; ctx.strokeRect(lx, ly, lw, lh);
    ctx.fillText(DL.fmt(mx, 2), lx + 3, ly + 12);
    ctx.strokeStyle = col.accent; ctx.lineWidth = 2; ctx.beginPath();
    var m = losses.length, stride = Math.max(1, Math.floor(m / 150));
    for (k = 0; k < m; k += stride) { var X1 = lx + (m < 2 ? 0 : k / (m - 1)) * lw, Y1 = ly + lh - Math.min(1, losses[k] / mx) * lh; k ? ctx.lineTo(X1, Y1) : ctx.moveTo(X1, Y1); }
    ctx.stroke();
    var nparams = 0; for (i = 0; i < net.W.length; i++) nparams += net.W[i].length * net.W[i][0].length + net.b[i].length;
    say('epoch <b>' + epoch + '</b> &nbsp; loss <b>' + DL.fmt(losses[losses.length - 1], 4) + '</b> &nbsp; accuracy <b>' + DL.fmt(100 * acc, 1) + '%</b> &nbsp; parameters <b>' + nparams + '</b> &nbsp; (initial loss should be near ln 2 = 0.693)');
  }
  c.draw = draw;
  makeData(); init();
});
