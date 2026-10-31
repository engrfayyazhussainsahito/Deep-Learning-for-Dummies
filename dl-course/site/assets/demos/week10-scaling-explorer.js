DL.register('week10-scaling-explorer', function (root) {
  var E = 1.69, A = 406.4, B = 410.7, al = 0.34, be = 0.28;
  function loss(N, D) { return E + A / Math.pow(N, al) + B / Math.pow(D, be); }
  var G = Math.pow(al * A / (be * B), 1 / (al + be)), a = be / (al + be);
  function nopt(C) { return G * Math.pow(C / 6, a); }
  var ctrl = DL.controls(root);
  var preset = DL.select(ctrl, { label: 'Preset', value: 'custom', options: [
    { value: 'custom', label: 'Custom (use sliders)' }, { value: 'gopher', label: 'Gopher: 280B params, 300B tokens' },
    { value: 'gpt3', label: 'GPT-3: 175B params, 300B tokens' }, { value: 'chin', label: 'Chinchilla: 70B params, 1.4T tokens' }] }, function (v) {
    var pr = { gopher: [280e9, 300e9], gpt3: [175e9, 300e9], chin: [70e9, 1.4e12] }[v];
    if (pr) { lc.set(Math.log10(6 * pr[0] * pr[1])); ln.set(Math.log10(pr[0])); }
    draw();
  });
  var lc = DL.slider(ctrl, { label: 'log10 compute C (FLOPs)', min: 18, max: 26, step: 0.05, value: 23.76, format: function (v) { return (+v).toFixed(2); } }, function () { preset.set('custom'); draw(); });
  var ln = DL.slider(ctrl, { label: 'log10 parameters N', min: 7, max: 12.5, step: 0.02, value: 10.85, format: function (v) { return (+v).toFixed(2); } }, function () { preset.set('custom'); draw(); });
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Model: L(N, D) = 1.69 + 406.4/N^0.34 + 410.7/D^0.28 (Hoffmann et al. 2022, approach 3), with D = C/(6N). Left: loss along the isoFLOP curve for the chosen budget; the green dot is the minimum. Right: compute-optimal N and D versus C; the vertical line marks your budget. Try the presets: Gopher and Chinchilla use the same compute but Chinchilla sits near the minimum.' }));
  function fmtE(x) { var e = Math.floor(Math.log10(x)); return (x / Math.pow(10, e)).toFixed(2) + 'e' + e; }
  function draw() {
    var ctx = c.ctx, col = DL.colors(), logC = lc.get(), C = Math.pow(10, logC), logN = ln.get(), N = Math.pow(10, logN), D = C / (6 * N);
    var No = nopt(C), Do = C / (6 * No), Lo = loss(No, Do), Lcur = loss(N, D);
    ctx.clearRect(0, 0, c.w, c.h);
    var half = Math.floor(c.w * 0.58);
    var P = DL.plot(c, { xmin: 7, xmax: 12.5, ymin: 1.8, ymax: 3.4, pad: { l: 46, r: c.w - half + 6, t: 14, b: 38 }, xlabel: 'log10 N  (D = C / 6N)', ylabel: 'predicted loss' });
    P.grid(); P.axes();
    var xs = DL.linspace(7, 12.5, 160), ys = xs.map(function (x) { var n = Math.pow(10, x); return loss(n, C / (6 * n)); });
    P.line(xs, ys, col.accent, 2.6);
    P.hline(E, col.muted, [2, 4]); P.text('E = 1.69', 7.1, E + 0.05, col.muted, 'left');
    P.points([Math.log10(No)], [Math.min(Lo, 3.4)], col.green, 6);
    P.points([logN], [Math.min(Math.max(Lcur, 1.8), 3.4)], col.amber, 6);
    var Q = DL.plot(c, { xmin: 18, xmax: 26, ymin: 7, ymax: 14, pad: { l: half + 46, r: 12, t: 14, b: 38 }, xlabel: 'log10 C', ylabel: 'log10 (N* or D*)' });
    Q.grid(); Q.axes();
    var cs = DL.linspace(18, 26, 40);
    Q.line(cs, cs.map(function (x) { return Math.log10(nopt(Math.pow(10, x))); }), col.green, 2.4);
    Q.line(cs, cs.map(function (x) { var cc = Math.pow(10, x); return Math.log10(cc / (6 * nopt(cc))); }), col.purple, 2.4);
    Q.text('D*', 18.3, 12.6, col.purple, 'left'); Q.text('N*', 18.3, 8.2, col.green, 'left');
    Q.vline(logC, col.amber);
    say('N = <b>' + fmtE(N) + '</b>, D = <b>' + fmtE(D) + '</b> tokens (' + DL.fmt(D / N, 1) + ' per param), loss = <b>' + DL.fmt(Lcur, 3) + '</b><br>' +
      'Compute-optimal at this C: N* = <b>' + fmtE(No) + '</b>, D* = <b>' + fmtE(Do) + '</b> (' + DL.fmt(Do / No, 1) + ' per param), L* = <b>' + DL.fmt(Lo, 3) + '</b>. Excess loss of your choice: <b>' + DL.fmt(Lcur - Lo, 3) + '</b> nats');
  }
  c.draw = draw;
});
