DL.register('week01-entropy-kl', function (root) {
  var K = 5, EPS = 1e-4;
  var p = [0.1, 0.4, 0.3, 0.15, 0.05], q = [0.2, 0.2, 0.2, 0.2, 0.2];
  var unit = 'bits';
  var ctrl = DL.controls(root);
  var row = DL.row(root);
  DL.select(ctrl, { label: 'Unit', value: 'bits', options: [{ value: 'bits', label: 'bits (log base 2)' }, { value: 'nats', label: 'nats (natural log)' }] }, function (v) { unit = v; draw(); });
  DL.button(row, 'q = uniform', function () { q = fill(1 / K); draw(); });
  DL.button(row, 'q = p', function () { q = p.slice(); draw(); });
  DL.button(row, 'p = peaked', function () { p = [0.9, 0.04, 0.03, 0.02, 0.01]; draw(); });
  DL.button(row, 'q misses p\'s mode', function () { p = [0.1, 0.6, 0.2, 0.06, 0.04]; q = [0.45, 0.01, 0.2, 0.14, 0.2]; norm(q); draw(); });
  DL.button(row, 'random', function () { p = rnd(); q = rnd(); draw(); });
  var c = DL.canvas(root, { aspect: 0.62 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Drag any bar (left bar of a pair = p, right bar = q); the other bars rescale to keep the sum at 1. Try: make q tiny where p is large (KL(p||q) explodes) and compare with KL(q||p); set q = p (KL = 0); make p one-hot-like (H(p) near 0, cross-entropy = -log q of the true class).' }));
  var rng = DL.rng(7);
  function fill(v) { var a = []; for (var i = 0; i < K; i++) a.push(v); return a; }
  function norm(a) { var s = DL.sum(a); for (var i = 0; i < a.length; i++) a[i] /= s; }
  function rnd() { var a = []; for (var i = 0; i < K; i++) a.push(0.05 + Math.pow(rng(), 2)); norm(a); return a; }
  function lg(x) { return unit === 'bits' ? Math.log(x) / Math.LN2 : Math.log(x); }
  function setProb(a, i, v) {
    v = DL.clamp(v, EPS, 1 - EPS * (K - 1));
    var others = 0, j; for (j = 0; j < K; j++) if (j !== i) others += a[j];
    var rest = 1 - v;
    for (j = 0; j < K; j++) if (j !== i) a[j] = others > 0 ? Math.max(EPS, a[j] * rest / others) : rest / (K - 1);
    a[i] = v; norm(a);
  }
  var geo = {};
  function layout() {
    geo.l = 44; geo.r = c.w - 12; geo.t = 22; geo.b1 = c.h * 0.58; geo.t2 = c.h * 0.66; geo.b2 = c.h - 22;
    geo.slot = (geo.r - geo.l) / K;
  }
  function yTop(v) { return geo.b1 - v * (geo.b1 - geo.t); }
  var active = null;
  function hit(x, y) {
    if (y > geo.b1 + 6 || y < geo.t - 10) return null;
    var i = Math.floor((x - geo.l) / geo.slot); if (i < 0 || i >= K) return null;
    var within = (x - geo.l) / geo.slot - i;
    return { i: i, arr: within < 0.5 ? 'p' : 'q' };
  }
  function apply(x, y) {
    if (!active) return;
    var v = (geo.b1 - y) / (geo.b1 - geo.t);
    setProb(active.arr === 'p' ? p : q, active.i, v);
    draw();
  }
  DL.drag(c.canvas, {
    down: function (x, y) { layout(); active = hit(x, y); apply(x, y); },
    move: function (x, y) { apply(x, y); },
    up: function () { active = null; }
  });
  function draw() {
    if (!c.w) return;
    layout();
    var col = DL.colors(), ctx = c.ctx, i;
    ctx.clearRect(0, 0, c.w, c.h);
    var mono = getComputedStyle(document.body).getPropertyValue('--font-mono'), sans = getComputedStyle(document.body).getPropertyValue('--font-sans');
    // top panel
    ctx.save();
    ctx.strokeStyle = col.line; ctx.fillStyle = col.muted; ctx.font = '11px ' + mono; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) { ctx.beginPath(); ctx.moveTo(geo.l, yTop(v)); ctx.lineTo(geo.r, yTop(v)); ctx.stroke(); ctx.fillText(v.toFixed(2), geo.l - 6, yTop(v)); });
    ctx.font = '12px ' + sans; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillStyle = col.blue; ctx.fillRect(geo.l + 2, 4, 10, 10); ctx.fillStyle = col.ink; ctx.fillText('p (true)', geo.l + 16, 3);
    ctx.fillStyle = col.accent; ctx.fillRect(geo.l + 90, 4, 10, 10); ctx.fillStyle = col.ink; ctx.fillText('q (model)', geo.l + 104, 3);
    ctx.textAlign = 'center';
    for (i = 0; i < K; i++) {
      var x0 = geo.l + i * geo.slot, bw = geo.slot * 0.38;
      ctx.fillStyle = col.blue; ctx.fillRect(x0 + geo.slot * 0.08, yTop(p[i]), bw, geo.b1 - yTop(p[i]));
      ctx.fillStyle = col.accent; ctx.fillRect(x0 + geo.slot * 0.08 + bw + 2, yTop(q[i]), bw, geo.b1 - yTop(q[i]));
      ctx.fillStyle = col.muted; ctx.font = '11px ' + mono; ctx.textBaseline = 'top';
      ctx.fillText('class ' + i, x0 + geo.slot / 2, geo.b1 + 4);
      ctx.textBaseline = 'bottom'; ctx.fillStyle = col.ink;
      ctx.fillText(p[i].toFixed(2), x0 + geo.slot * 0.08 + bw / 2, yTop(p[i]) - 1);
      ctx.fillText(q[i].toFixed(2), x0 + geo.slot * 0.08 + bw * 1.5 + 2, yTop(q[i]) - 1);
    }
    ctx.restore();
    // bottom panel: per-class terms p_i log(p_i/q_i)
    var terms = p.map(function (v, k) { return v * lg(v / q[k]); });
    var rterms = q.map(function (v, k) { return v * lg(v / p[k]); });
    var mx = Math.max(0.2, Math.max.apply(null, terms.map(Math.abs)), Math.max.apply(null, rterms.map(Math.abs)));
    var zero = (geo.t2 + geo.b2) / 2, sc = (geo.b2 - geo.t2) / 2 / mx * 0.92;
    ctx.save();
    ctx.strokeStyle = col.muted; ctx.beginPath(); ctx.moveTo(geo.l, zero); ctx.lineTo(geo.r, zero); ctx.stroke();
    ctx.font = '12px ' + sans; ctx.fillStyle = col.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('per-class terms of KL(p||q) = Σ pᵢ log(pᵢ/qᵢ) (green: positive, red: negative); sum is always ≥ 0', geo.l, geo.t2 - 14);
    for (i = 0; i < K; i++) {
      var x1 = geo.l + i * geo.slot + geo.slot * 0.2, w = geo.slot * 0.6, h = terms[i] * sc;
      ctx.fillStyle = terms[i] >= 0 ? col.green : col.red;
      ctx.fillRect(x1, h >= 0 ? zero - h : zero, w, Math.abs(h));
      ctx.fillStyle = col.ink; ctx.font = '11px ' + mono; ctx.textAlign = 'center'; ctx.textBaseline = h >= 0 ? 'bottom' : 'top';
      ctx.fillText(terms[i].toFixed(3), x1 + w / 2, h >= 0 ? zero - h - 1 : zero - h + 1);
    }
    ctx.restore();
    var Hp = 0, Hpq = 0, Hq = 0, kl = 0, klr = 0, js = 0;
    for (i = 0; i < K; i++) {
      Hp -= p[i] * lg(p[i]); Hq -= q[i] * lg(q[i]); Hpq -= p[i] * lg(q[i]);
      kl += p[i] * lg(p[i] / q[i]); klr += q[i] * lg(q[i] / p[i]);
      var m = (p[i] + q[i]) / 2; js += 0.5 * p[i] * lg(p[i] / m) + 0.5 * q[i] * lg(q[i] / m);
    }
    var u = unit === 'bits' ? ' bits' : ' nats';
    say('H(p) = <b>' + DL.fmt(Hp, 3) + '</b> &nbsp; H(q) = ' + DL.fmt(Hq, 3) + ' &nbsp; cross-entropy H(p,q) = <b>' + DL.fmt(Hpq, 3) + '</b>' + u + '<br>' +
      'KL(p||q) = <b>' + DL.fmt(Math.max(0, kl), 3) + '</b> = H(p,q) − H(p) = ' + DL.fmt(Hpq - Hp, 3) + ' &nbsp; KL(q||p) = <b>' + DL.fmt(Math.max(0, klr), 3) + '</b> &nbsp; Jensen-Shannon = ' + DL.fmt(Math.max(0, js), 3) + '<br>' +
      'max possible H = log K = ' + DL.fmt(lg(K), 3) + u + (Math.abs(kl - klr) > 0.05 ? ' &nbsp; (note KL is asymmetric)' : ''));
  }
  c.draw = draw;
});
