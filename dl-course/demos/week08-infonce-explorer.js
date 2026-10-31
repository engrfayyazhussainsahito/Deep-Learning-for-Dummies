DL.register('week08-infonce-explorer', function (root) {
  var ctrl = DL.controls(root);
  var tau = DL.slider(ctrl, { label: 'Temperature τ', min: 0.03, max: 1.5, step: 0.01, value: 0.2 }, draw);
  var sp = DL.slider(ctrl, { label: 'Positive cosine s⁺', min: -1, max: 1, step: 0.02, value: 0.6 }, draw);
  var K = DL.slider(ctrl, { label: 'Negatives K', min: 1, max: 63, step: 1, value: 7 }, draw);
  var ctrl2 = DL.controls(root);
  var hard = DL.slider(ctrl2, { label: 'Negative hardness (mean cosine)', min: -0.4, max: 0.9, step: 0.02, value: 0 }, draw);
  var margin = DL.slider(ctrl2, { label: 'Triplet margin', min: 0, max: 1.5, step: 0.05, value: 0.5 }, draw);
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Left curve: the InfoNCE loss as a function of the positive similarity s⁺, with K negatives all sitting at the chosen mean cosine. A random (uninformative) model gets loss log(K+1); the loss goes to 0 only when s⁺ exceeds the negatives by many multiples of τ. Small τ sharpens the contrast and focuses gradient on hard negatives; large τ flattens everything. Right: the triplet hinge for comparison. Try making the negatives hard (cosine near s⁺) and watch the gradient grow.' }));
  function nce(s, t, k, hn) { // exact loss with all K negatives at cosine hn
    var a = s / t, b = Math.log(k) + hn / t, m = Math.max(a, b);
    return -(a - (m + Math.log(Math.exp(a - m) + Math.exp(b - m))));
  }
  function draw() {
    var col = DL.colors(), ctx = c.ctx, t = tau.get(), k = Math.round(K.get()), hn = hard.get(), s0 = sp.get(), mg = margin.get();
    var half = Math.floor(c.w / 2);
    // left: InfoNCE vs s+
    var L = DL.plot(c, { xmin: -1, xmax: 1, ymin: 0, ymax: Math.max(1, Math.log(k + 1) * 1.6), xlabel: 'positive cosine s⁺', ylabel: 'loss', pad: { l: 44, r: c.w - half + 10, t: 14, b: 34 } });
    L.clear(); L.grid(); L.axes();
    L.fn(function (s) { return Math.min(nce(s, t, k, hn), Math.log(k + 1) * 1.6); }, col.accent, 2.5);
    L.hline && L.hline(Math.log(k + 1), col.muted);
    ctx.save(); ctx.fillStyle = col.muted; ctx.font = '11px ' + getComputedStyle(document.body).getPropertyValue('--font-mono'); ctx.textAlign = 'left';
    ctx.fillText('log(K+1) = ' + DL.fmt(Math.log(k + 1), 2), L.px(-0.95), L.py(Math.log(k + 1)) - 6);
    ctx.fillStyle = col.accent; ctx.beginPath(); ctx.arc(L.px(s0), L.py(nce(s0, t, k, hn)), 5, 0, 6.2832); ctx.fill();
    ctx.restore();
    // right: triplet hinge vs d(a,n) - d(a,p)
    var R = DL.plot(c, { xmin: -1.5, xmax: 1.5, ymin: 0, ymax: 2, xlabel: 'd(a,n) - d(a,p)', ylabel: 'hinge loss', pad: { l: half + 44, r: 14, t: 14, b: 34 } });
    R.axes && 0;
    R.grid(); R.axes();
    R.fn(function (x) { return Math.max(0, mg - x); }, col.blue, 2.5);
    var gap = (1 - hn) - (1 - s0); // cosine distances: d = 1 - cos
    ctx.save(); ctx.fillStyle = col.blue; ctx.beginPath(); ctx.arc(R.px(DL.clamp(gap, -1.5, 1.5)), R.py(Math.max(0, mg - gap)), 5, 0, 6.2832); ctx.fill(); ctx.restore();
    // gradient wrt s+ of InfoNCE: -(1 - p+)/t
    var a = s0 / t, b = Math.log(k) + hn / t, m = Math.max(a, b);
    var pPos = Math.exp(a - m) / (Math.exp(a - m) + Math.exp(b - m));
    say('InfoNCE = <b>' + DL.fmt(nce(s0, t, k, hn), 3) + '</b> (chance level ' + DL.fmt(Math.log(k + 1), 3) + ')   p(positive) = <b>' + DL.fmt(pPos, 3) + '</b>   ∂loss/∂s⁺ = <b>' + DL.fmt(-(1 - pPos) / t, 2) + '</b><br>triplet: d(a,p) = ' + DL.fmt(1 - s0, 2) + ', d(a,n) = ' + DL.fmt(1 - hn, 2) + ', hinge = <b>' + DL.fmt(Math.max(0, mg - gap), 3) + '</b> ' + (gap >= mg ? '(satisfied, zero gradient)' : '(violated, active)'));
  }
  c.draw = draw;
  draw();
});
