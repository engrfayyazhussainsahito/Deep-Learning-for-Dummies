DL.register('week15-ppo-clip', function (root) {
  // Left: L_CLIP(r) for one sample. Right: a one-state, two-action policy p = sigmoid(theta) trained by
  // repeated gradient ascent on the same batch (exact expectation under the OLD policy), clipped vs unclipped.
  var ctrl = DL.controls(root);
  var adv = DL.slider(ctrl, { label: 'Advantage A', min: -2, max: 2, step: 0.1, value: 1 }, draw);
  var eps = DL.slider(ctrl, { label: 'Clip ε', min: 0.05, max: 0.5, step: 0.05, value: 0.2 }, draw);
  var rr = DL.slider(ctrl, { label: 'Ratio r (marker)', min: 0, max: 2.5, step: 0.01, value: 1.1 }, draw);
  var lr = DL.slider(ctrl, { label: 'Step size', min: 0.05, max: 2, step: 0.05, value: 0.5 }, draw);
  var ep = DL.slider(ctrl, { label: 'Epochs K', min: 1, max: 60, step: 1, value: 30, format: function (v) { return String(Math.round(v)); } }, draw);
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Left: the clipped objective for one sample (solid) vs the plain surrogate r·A (dashed); the red part has zero gradient. Right: a two-action policy where action 1 has advantage +A and action 0 has −A; both policies start at p = 0.5 and take K gradient steps on the same batch. The unclipped surrogate keeps pushing p toward 1; PPO-Clip stops once the probability of the good action has changed by about ε (ratio 1+ε). Try large step sizes and many epochs.' }));

  function sig(x) { return 1 / (1 + Math.exp(-x)); }
  function clip(x, a, b) { return Math.max(a, Math.min(b, x)); }
  function Lclip(r, A, e) { return Math.min(r * A, clip(r, 1 - e, 1 + e) * A); }
  // expected objective for policy parameter th, old p0 = 0.5, A1 = +A, A0 = -A
  function obj(th, A, e, useClip) {
    var p = sig(th), p0 = 0.5, r1 = p / p0, r0 = (1 - p) / (1 - p0);
    if (useClip) return p0 * Lclip(r1, A, e) + (1 - p0) * Lclip(r0, -A, e);
    return p0 * r1 * A + (1 - p0) * r0 * (-A);
  }
  function run(useClip) {
    var th = 0, h = 1e-5, A = adv.get(), e = eps.get(), path = [sig(th)], K = Math.round(ep.get()), s = lr.get();
    for (var k = 0; k < K; k++) {
      var g = (obj(th + h, A, e, useClip) - obj(th - h, A, e, useClip)) / (2 * h);
      th += s * g; th = clip(th, -30, 30); path.push(sig(th));
    }
    return path;
  }
  function draw() {
    if (!c.w) return;
    var ctx = c.ctx, col = DL.colors(), W = c.w, H = c.h, half = W / 2, A = adv.get(), e = eps.get(), r = rr.get(), i;
    ctx.clearRect(0, 0, W, H);
    var vals = [0, 2.5].map(function (x) { return x * A; }); var ymin = Math.min(-0.1, vals[0], vals[1]) - 0.2, ymax = Math.max(0.1, vals[0], vals[1]) + 0.2;
    var P1 = DL.plot({ ctx: ctx, w: half, h: H }, { xmin: 0, xmax: 2.5, ymin: ymin, ymax: ymax, xlabel: 'ratio r', ylabel: 'objective', pad: { l: 44, r: 10, t: 14, b: 34 } });
    P1.grid(); P1.axes();
    P1.vline(1, col.muted); P1.vline(1 - e, col.amber); P1.vline(1 + e, col.amber);
    P1.fn(function (x) { return x * A; }, col.muted, 1.5, [5, 4]);
    // clipped objective, drawing zero-gradient pieces in red
    var xs = DL.linspace(0, 2.5, 120), flatX = [], flatY = [], liveX = [], liveY = [];
    xs.forEach(function (x) {
      var flat = (A > 0 && x > 1 + e) || (A < 0 && x < 1 - e) || A === 0;
      (flat ? flatX : liveX).push(x); (flat ? flatY : liveY).push(Lclip(x, A, e));
    });
    P1.points(liveX, liveY, col.accent, 2); P1.points(flatX, flatY, col.red, 2);
    P1.points([r], [Lclip(r, A, e)], col.ink, 5);
    var grad = ((A > 0 && r > 1 + e) || (A < 0 && r < 1 - e) || A === 0) ? 0 : A;
    // right panel
    var pu = run(false), pc = run(true), K = pu.length - 1;
    var P2 = DL.plot({ ctx: (function () { ctx.save(); ctx.translate(half, 0); return ctx; })(), w: half, h: H }, { xmin: 0, xmax: K, ymin: 0, ymax: 1, xlabel: 'gradient step', ylabel: 'p(good action)', pad: { l: 44, r: 10, t: 14, b: 34 } });
    P2.grid(); P2.axes();
    var kx = DL.range(K + 1);
    P2.line(kx, pu, col.blue, 2.5); P2.line(kx, pc, col.accent, 2.5);
    var pOld = 0.5, hi = pOld * (1 + e), lo = pOld * (1 - e), goodIsAdv = A >= 0;
    var band = goodIsAdv ? hi : lo; P2.hline(Math.min(1, Math.max(0, band)), col.amber);
    ctx.fillStyle = col.blue; ctx.font = '12px sans-serif'; ctx.textAlign = 'left'; ctx.fillText('unclipped', P2.px(K * 0.55), P2.py(0.5) + 40);
    ctx.fillStyle = col.accent; ctx.fillText('PPO-Clip', P2.px(K * 0.55), P2.py(0.5) + 56);
    ctx.restore();
    var fu = pu[K], fc = pc[K];
    say('marker: r = <b>' + DL.fmt(r, 2) + '</b>, L = <b>' + DL.fmt(Lclip(r, A, e), 3) + '</b>, dL/dr = <b>' + DL.fmt(grad, 2) + '</b>' + (grad === 0 && A !== 0 ? ' <span style="color:' + col.red + '">(clipped: no gradient)</span>' : '') +
      ' &nbsp; | &nbsp; after K steps: p<sub>unclipped</sub> = <b>' + DL.fmt(fu, 3) + '</b> (ratio ' + DL.fmt(fu / 0.5, 2) + '), p<sub>clip</sub> = <b>' + DL.fmt(fc, 3) + '</b> (ratio ' + DL.fmt(fc / 0.5, 2) + ')');
  }
  c.draw = draw;
});
