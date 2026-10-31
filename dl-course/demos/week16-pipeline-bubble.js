DL.register('week16-pipeline-bubble', function (root) {
  var ctrl = DL.controls(root);
  var P = DL.slider(ctrl, { label: 'Stages p', min: 2, max: 8, step: 1, value: 4, format: function (v) { return String(Math.round(v)); } }, draw);
  var M = DL.slider(ctrl, { label: 'Micro-batches m', min: 1, max: 24, step: 1, value: 8, format: function (v) { return String(Math.round(v)); } }, draw);
  var R = DL.slider(ctrl, { label: 'Backward / forward cost', min: 1, max: 3, step: 0.5, value: 2, format: function (v) { return DL.fmt(v, 1) + '×'; } }, draw);
  var sch = DL.select(ctrl, { label: 'Schedule', value: 'gpipe', options: [{ value: 'gpipe', label: 'GPipe (all F, then all B)' }, { value: '1f1b', label: '1F1B (PipeDream-Flush style)' }] }, draw);
  var c = DL.canvas(root, { aspect: 0.5 });
  var say = DL.readout(root);
  root.appendChild(DL.el('div', { class: 'dl-hint', text: 'Blue = forward, green = backward, numbers = micro-batch index; grey is idle time (the bubble). More micro-batches shrink the bubble toward (p-1)/(m+p-1). Compare peak stored activations: GPipe holds all m micro-batches on the first stage, 1F1B holds at most p.' }));

  function simulate(p, m, r, kind) {
    var ops = [], s, j;
    for (s = 0; s < p; s++) {
      var list = [];
      if (kind === 'gpipe') { for (j = 0; j < m; j++) list.push(['F', j]); for (j = 0; j < m; j++) list.push(['B', j]); }
      else { var w = Math.min(p - s - 1, m); for (j = 0; j < w; j++) list.push(['F', j]); for (j = 0; j < m - w; j++) { list.push(['F', w + j]); list.push(['B', j]); } for (j = m - w; j < m; j++) list.push(['B', j]); }
      ops.push(list);
    }
    var fin = {}, free = [], ptr = [], placed = [], progress = true, total = 2 * m * p, done = 0;
    for (s = 0; s < p; s++) { free.push(0); ptr.push(0); placed.push([]); }
    while (done < total && progress) {
      progress = false;
      for (s = 0; s < p; s++) {
        if (ptr[s] >= ops[s].length) continue;
        var op = ops[s][ptr[s]], t, dep;
        if (op[0] === 'F') dep = s === 0 ? 0 : fin['F' + (s - 1) + '_' + op[1]];
        else dep = s === p - 1 ? fin['F' + s + '_' + op[1]] : fin['B' + (s + 1) + '_' + op[1]];
        if (dep === undefined) continue;
        var d = op[0] === 'F' ? 1 : r; t = Math.max(free[s], dep);
        fin[op[0] + s + '_' + op[1]] = t + d; free[s] = t + d; placed[s].push({ k: op[0], j: op[1], t: t, d: d }); ptr[s]++; done++; progress = true;
      }
    }
    var span = Math.max.apply(null, free), peak = 0;
    for (s = 0; s < p; s++) { var cur = 0; placed[s].forEach(function (o) { cur += o.k === 'F' ? 1 : -1; if (cur > peak) peak = cur; }); }
    return { placed: placed, span: span, peak: peak };
  }
  function draw() {
    if (!c.w) return;
    var p = Math.round(P.get()), m = Math.round(M.get()), r = R.get(), res = simulate(p, m, r, sch.get());
    var ctx = c.ctx, col = DL.colors(), W = c.w, H = c.h, pl = 56, pr = 10, pt = 10, pb = 26;
    ctx.clearRect(0, 0, W, H);
    var rowH = (H - pt - pb) / p, sx = (W - pl - pr) / res.span;
    ctx.font = '11px sans-serif'; ctx.textBaseline = 'middle';
    for (var s = 0; s < p; s++) {
      var y = pt + s * rowH;
      ctx.fillStyle = col.soft; ctx.fillRect(pl, y + 2, W - pl - pr, rowH - 4);
      ctx.fillStyle = col.muted; ctx.textAlign = 'right'; ctx.fillText('stage ' + s, pl - 6, y + rowH / 2);
      res.placed[s].forEach(function (o) {
        ctx.fillStyle = o.k === 'F' ? col.blue : col.green; ctx.globalAlpha = 0.85;
        ctx.fillRect(pl + o.t * sx + 0.5, y + 3, Math.max(1, o.d * sx - 1), rowH - 6); ctx.globalAlpha = 1;
        if (o.d * sx > 14) { ctx.fillStyle = col.bg; ctx.textAlign = 'center'; ctx.fillText(String(o.j), pl + (o.t + o.d / 2) * sx, y + rowH / 2); }
      });
    }
    ctx.fillStyle = col.muted; ctx.textAlign = 'center'; ctx.fillText('time (units of one micro-batch forward)', (pl + W - pr) / 2, H - 10);
    var ideal = m * (1 + r), bubble = 1 - ideal / res.span, formula = (p - 1) / (m + p - 1);
    say('makespan: <b>' + DL.fmt(res.span, 1) + '</b> &nbsp; ideal (no bubble): ' + DL.fmt(ideal, 1) + ' &nbsp; bubble: <b>' + DL.fmt(100 * bubble, 1) + '%</b> &nbsp; formula (p-1)/(m+p-1) = ' + DL.fmt(100 * formula, 1) + '% &nbsp; peak stored micro-batch activations on a stage: <b>' + res.peak + '</b>');
  }
  c.draw = draw;
  draw();
});
