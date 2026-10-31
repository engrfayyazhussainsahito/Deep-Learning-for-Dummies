DL.register('week12-vae-elbo', function (root) {
  // Linear-Gaussian VAE in 1-D, fully analytic.
  // data x ~ N(0, T^2); prior z ~ N(0,1); encoder q(z|x)=N(a x, s^2); decoder p(x|z)=N(w z, sig2)
  var T = 2;
  var ctrl = DL.controls(root), row = DL.row(root);
  var aS = DL.slider(ctrl, { label: 'Encoder gain a', min: -1.5, max: 1.5, step: 0.01, value: 0.3 }, draw);
  var sS = DL.slider(ctrl, { label: 'Encoder std s', min: 0.05, max: 2, step: 0.01, value: 0.9 }, draw);
  var wS = DL.slider(ctrl, { label: 'Decoder gain w', min: -4, max: 4, step: 0.01, value: 0.8 }, draw);
  var vS = DL.slider(ctrl, { label: 'Decoder noise σ²', min: 0.05, max: 4, step: 0.01, value: 0.5 }, draw);
  var bS = DL.slider(ctrl, { label: 'β (KL weight)', min: 0.05, max: 10, step: 0.05, value: 1 }, draw);
  var xS = DL.slider(ctrl, { label: 'Example x', min: -4, max: 4, step: 0.05, value: 2 }, draw);
  DL.button(row, 'Optimise a, s, w (300 Adam steps)', optimise);
  DL.button(row, 'Reset', function () { aS.set(0.3); sS.set(0.9); wS.set(0.8); draw(); });
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Data are N(0, 2²). Left: the prior (grey), the aggregate posterior over all x (accent), and the posterior of the chosen example x (blue). Right: expected reconstruction term and β·KL, averaged over the data. Press Optimise at β = 1 (a latent code with KL about 1 nat is learned), then set β = 10, or keep β = 1 and raise the decoder noise σ² to 4, and optimise again: the encoder gain a goes to 0, the posterior equals the prior, KL = 0 and the decoder ignores z (posterior collapse). Small β gives a sharper reconstruction at the price of a latent code far from the prior.' }));
  function terms(a, s, w, v) {
    var T2 = T * T;
    var rec = 0.5 * Math.log(2 * Math.PI * v) + ((1 - w * a) * (1 - w * a) * T2 + w * w * s * s) / (2 * v);
    var kl = 0.5 * (a * a * T2 + s * s - 1 - Math.log(s * s));
    return { rec: rec, kl: kl };
  }
  function lossOf(p, beta, v) { var t = terms(p[0], Math.exp(p[1]), p[2], v); return t.rec + beta * t.kl; }
  function optimise() {
    var p = [aS.get(), Math.log(sS.get()), wS.get()], m = [0, 0, 0], u = [0, 0, 0], beta = bS.get(), v = vS.get(), h = 1e-5, lr = 0.03;
    for (var t = 1; t <= 300; t++) {
      var g = [];
      for (var i = 0; i < 3; i++) { var pp = p.slice(), pm = p.slice(); pp[i] += h; pm[i] -= h; g.push((lossOf(pp, beta, v) - lossOf(pm, beta, v)) / (2 * h)); }
      for (i = 0; i < 3; i++) {
        m[i] = 0.9 * m[i] + 0.1 * g[i]; u[i] = 0.999 * u[i] + 0.001 * g[i] * g[i];
        p[i] -= lr * (m[i] / (1 - Math.pow(0.9, t))) / (Math.sqrt(u[i] / (1 - Math.pow(0.999, t))) + 1e-8);
      }
    }
    aS.set(DL.clamp(p[0], -1.5, 1.5)); sS.set(DL.clamp(Math.exp(p[1]), 0.05, 2)); wS.set(DL.clamp(p[2], -4, 4)); draw();
  }
  function gauss(x, m, s) { return Math.exp(-0.5 * (x - m) * (x - m) / (s * s)) / (s * 2.5066282746); }
  function draw() {
    if (!c.w) return;
    var col = DL.colors(), ctx = c.ctx, a = aS.get(), s = sS.get(), w = wS.get(), v = vS.get(), beta = bS.get(), x = xS.get();
    var t = terms(a, s, w, v), total = t.rec + beta * t.kl;
    ctx.clearRect(0, 0, c.w, c.h);
    var lw = c.w * 0.62;
    var P = DL.plot(c, { xmin: -4, xmax: 4, ymin: 0, ymax: 1.6, xlabel: 'z', ylabel: 'density', pad: { l: 44, r: c.w - lw + 6, t: 14, b: 34 } });
    P.grid(); P.axes();
    var aggS = Math.sqrt(a * a * T * T + s * s);
    P.fn(function (z) { return gauss(z, 0, 1); }, col.muted, 2);
    P.fn(function (z) { return Math.min(1.6, gauss(z, 0, aggS)); }, col.accent, 2.5);
    P.fn(function (z) { return Math.min(1.6, gauss(z, a * x, s)); }, col.blue, 2.5);
    P.text('prior N(0,1)', -3.9, 1.5, col.muted, 'left'); P.text('aggregate q(z)', -3.9, 1.38, col.accent, 'left'); P.text('q(z|x=' + DL.fmt(x, 1) + ')', -3.9, 1.26, col.blue, 'left');
    // bars
    var bx = lw + 26, bw = (c.w - bx - 20) / 3, top = 24, bot = c.h - 40;
    var vals = [t.rec, beta * t.kl, total], names = ['recon', 'β·KL', 'total'], cols = [col.green, col.purple, col.ink];
    var mx = Math.max(1, vals[0], vals[1], vals[2]) * 1.1;
    ctx.save(); ctx.font = '11px ' + getComputedStyle(document.body).getPropertyValue('--font-mono'); ctx.textAlign = 'center';
    for (var i = 0; i < 3; i++) {
      var h = Math.max(0, vals[i]) / mx * (bot - top), x0 = bx + i * bw + 6;
      ctx.fillStyle = cols[i]; ctx.globalAlpha = 0.8; ctx.fillRect(x0, bot - h, bw - 12, h); ctx.globalAlpha = 1;
      ctx.fillStyle = col.ink; ctx.textBaseline = 'bottom'; ctx.fillText(DL.fmt(vals[i], 2), x0 + (bw - 12) / 2, bot - h - 2);
      ctx.textBaseline = 'top'; ctx.fillStyle = col.muted; ctx.fillText(names[i], x0 + (bw - 12) / 2, bot + 6);
    }
    ctx.strokeStyle = col.muted; ctx.beginPath(); ctx.moveTo(bx, bot); ctx.lineTo(c.w - 14, bot); ctx.stroke();
    ctx.fillStyle = col.ink; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('negative ELBO terms (nats)', bx, 4);
    ctx.restore();
    say('E[−log p(x|z)] = <b>' + DL.fmt(t.rec, 3) + '</b> &nbsp; KL(q(z|x)‖p(z)) averaged over x = <b>' + DL.fmt(t.kl, 3) + '</b> &nbsp; loss = rec + β·KL = <b>' + DL.fmt(total, 3) + '</b><br>' +
      'KL for the chosen x: ½(a²x² + s² − 1 − ln s²) = <b>' + DL.fmt(0.5 * (a * a * x * x + s * s - 1 - Math.log(s * s)), 3) + '</b> &nbsp; aggregate posterior std = ' + DL.fmt(aggS, 2) + ' (prior: 1.00)' + (Math.abs(a) < 0.03 && t.kl < 0.02 ? ' &nbsp; <b>posterior collapse</b>' : ''));
  }
  c.draw = draw;
});
