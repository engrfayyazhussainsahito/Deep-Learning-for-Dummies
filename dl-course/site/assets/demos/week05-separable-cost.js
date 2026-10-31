DL.register('week05-separable-cost', function (root) {
  var ctrl = DL.controls(root);
  var cin = DL.slider(ctrl, { label: 'C_in', min: 4, max: 512, step: 4, value: 64 }, draw);
  var cout = DL.slider(ctrl, { label: 'C_out', min: 4, max: 512, step: 4, value: 128 }, draw);
  var k = DL.slider(ctrl, { label: 'Kernel k', min: 1, max: 7, step: 2, value: 3 }, draw);
  var hw = DL.slider(ctrl, { label: 'Feature map H=W', min: 7, max: 224, step: 1, value: 56 }, draw);
  var grp = DL.slider(ctrl, { label: 'Groups (grouped conv)', min: 1, max: 64, step: 1, value: 1 }, draw);
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Compare five ways to map C_in to C_out channels. Try k=3 and C_out=128 (ratio about 1/9 + 1/128), then shrink C_out: the 1/k^2 term dominates. Bottleneck = 1x1 down to C/4, kxk, 1x1 up. Groups apply to the grouped-conv bar (must divide C_in and C_out; invalid values are rounded down to a divisor).' }));
  function gcdDiv(g, a, b) { while (g > 1 && (a % g !== 0 || b % g !== 0)) g--; return g; }
  function fmtN(v) { return v >= 1e9 ? (v / 1e9).toFixed(2) + 'G' : v >= 1e6 ? (v / 1e6).toFixed(2) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(1) + 'k' : String(Math.round(v)); }
  function draw() {
    if (!c.w) return;
    var ci = Math.round(cin.get()), co = Math.round(cout.get()), kk = Math.round(k.get()), s = Math.round(hw.get());
    var g = gcdDiv(Math.round(grp.get()), ci, co), mid = Math.max(1, Math.round(co / 4)), cm = Math.max(1, Math.round(ci / 4));
    var items = [
      ['standard kxk', kk * kk * ci * co, 'accent'],
      ['grouped (g=' + g + ')', kk * kk * ci * co / g, 'purple'],
      ['depthwise + 1x1', kk * kk * ci + ci * co, 'green'],
      ['bottleneck 1x1-kxk-1x1', ci * cm + kk * kk * cm * cm + cm * co, 'amber'],
      ['1x1 only', ci * co, 'blue']
    ];
    var col = DL.colors(), ctx = c.ctx;
    ctx.clearRect(0, 0, c.w, c.h);
    var left = 150, right = 110, top = 30, bh = (c.h - top - 20) / items.length, mx = Math.max.apply(null, items.map(function (i) { return i[1]; }));
    ctx.font = 'bold 12px sans-serif'; ctx.fillStyle = col.ink; ctx.textAlign = 'center';
    ctx.fillText('parameters (bar) and multiply-adds at ' + s + 'x' + s + ' (right)', c.w / 2, 14);
    items.forEach(function (it, i) {
      var y = top + i * bh, w = (c.w - left - right) * it[1] / mx;
      ctx.fillStyle = col.ink; ctx.font = '12px sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(it[0], left - 8, y + bh * 0.42);
      ctx.fillStyle = col[it[2]]; ctx.globalAlpha = 0.85; ctx.fillRect(left, y + 4, Math.max(w, 2), bh - 12); ctx.globalAlpha = 1;
      ctx.fillStyle = col.ink; ctx.textAlign = 'left';
      ctx.fillText(fmtN(it[1]), left + Math.max(w, 2) + 6 > c.w - right ? left + 4 : left + Math.max(w, 2) + 6, y + bh * 0.42);
      ctx.fillStyle = col.muted; ctx.textAlign = 'right';
      ctx.fillText(fmtN(it[1] * s * s) + ' MACs', c.w - 6, y + bh * 0.42);
    });
    var ratio = items[2][1] / items[0][1];
    say('Standard: <b>' + fmtN(items[0][1]) + '</b> params. Depthwise-separable: <b>' + fmtN(items[2][1]) + '</b> = ' + DL.fmt(100 * ratio, 1) + '% of standard (theory 1/C_out + 1/k² = ' + DL.fmt(1 / co + 1 / (kk * kk), 3) + '; ' + DL.fmt(1 / ratio, 1) + '× cheaper). Biases and BatchNorm parameters excluded. Same-padding assumed, so the spatial size does not change.');
  }
  c.draw = draw;
});
