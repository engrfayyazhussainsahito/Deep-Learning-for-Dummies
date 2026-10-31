DL.register('week02-activations', function (root) {
  function erf(x) { var s = x < 0 ? -1 : 1; x = Math.abs(x); var t = 1 / (1 + 0.3275911 * x); var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; }
  function Phi(x) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
  function phi(x) { return Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI); }
  var alpha = 0.1;
  var ACT = {
    sigmoid: { name: 'sigmoid', f: DL.sigmoid, d: function (x) { var s = DL.sigmoid(x); return s * (1 - s); } },
    tanh: { name: 'tanh', f: Math.tanh, d: function (x) { var t = Math.tanh(x); return 1 - t * t; } },
    relu: { name: 'ReLU', f: function (x) { return Math.max(0, x); }, d: function (x) { return x > 0 ? 1 : 0; } },
    leaky: { name: 'Leaky ReLU', f: function (x) { return x > 0 ? x : alpha * x; }, d: function (x) { return x > 0 ? 1 : alpha; } },
    gelu: { name: 'GELU (exact)', f: function (x) { return x * Phi(x); }, d: function (x) { return Phi(x) + x * phi(x); } },
    silu: { name: 'SiLU / Swish', f: function (x) { return x * DL.sigmoid(x); }, d: function (x) { var s = DL.sigmoid(x); return s * (1 + x * (1 - s)); } }
  };
  var ctrl = DL.controls(root);
  var sel = DL.select(ctrl, { label: 'Activation', value: 'sigmoid', options: Object.keys(ACT).map(function (k) { return { value: k, label: ACT[k].name }; }) }, draw);
  var xs = DL.slider(ctrl, { label: 'Probe input x', min: -6, max: 6, step: 0.05, value: 1.5, format: function (v) { return (+v).toFixed(2); } }, draw);
  var sg = DL.slider(ctrl, { label: 'Pre-activation std σ', min: 0.2, max: 4, step: 0.05, value: 1, format: function (v) { return (+v).toFixed(2); } }, draw);
  var dp = DL.slider(ctrl, { label: 'Depth L', min: 1, max: 50, step: 1, value: 20, format: function (v) { return String(Math.round(v)); } }, draw);
  var al = DL.slider(ctrl, { label: 'Leaky slope α', min: 0, max: 0.5, step: 0.01, value: 0.1, format: function (v) { return (+v).toFixed(2); } }, function (v) { alpha = v; draw(); });
  var c = DL.canvas(root, { aspect: 0.5 });
  var P = DL.plot(c, { xmin: -6, xmax: 6, ymin: -2, ymax: 3, xlabel: 'x', ylabel: 'value' });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Orange: f(x). Blue: f′(x). Move the probe to read both. The depth experiment multiplies the average derivative magnitude over L layers for pre-activations z ~ N(0, σ²): a crude proxy for how much gradient signal survives (real backprop also multiplies by weight matrices; see Week 3 on initialization). Try sigmoid: its derivative never exceeds 0.25. ReLU averages 0.5 per layer, which He initialization compensates with a factor of 2 in the weight variance.' }));
  function meanAbsD(a, s) { var sum = 0, wsum = 0, n = 161; for (var i = 0; i < n; i++) { var z = -4 + 8 * i / (n - 1), w = Math.exp(-z * z / 2); sum += w * Math.abs(a.d(z * s)); wsum += w; } return sum / wsum; }
  function draw() {
    if (!c.w) return;
    var a = ACT[sel.get()], col = DL.colors(), x = xs.get(), L = Math.round(dp.get()), s = sg.get();
    P.clear(); P.grid(); P.axes(); P.hline(0, col.line, [2, 3]);
    P.fn(a.d, col.blue, 2.2); P.fn(a.f, col.accent, 2.8);
    P.vline(x, col.muted);
    P.points([x, x], [a.f(x), a.d(x)], col.ink, 4.5);
    var m = meanAbsD(a, s), prod = Math.pow(m, L);
    say('f(' + DL.fmt(x, 2) + ') = <b>' + DL.fmt(a.f(x), 4) + '</b> &nbsp; f′(' + DL.fmt(x, 2) + ') = <b>' + DL.fmt(a.d(x), 4) + '</b> &nbsp;|&nbsp; E|f′(z)| at σ = ' + DL.fmt(s, 2) + ': <b>' + DL.fmt(m, 3) + '</b> &nbsp; over L = ' + L + ' layers: <b>' + (prod < 1e-3 ? prod.toExponential(2) : DL.fmt(prod, 4)) + '</b>');
  }
  c.draw = draw;
});
