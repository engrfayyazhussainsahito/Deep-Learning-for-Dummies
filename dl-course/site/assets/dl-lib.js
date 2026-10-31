/*
 * DL helper library for interactive demos.
 * Every demo file calls  DL.register('week05-conv-explorer', function (root) { ... })
 * where `root` is an empty <div class="demo-body"> to fill with controls + canvases.
 * All DOM helpers below produce elements styled by the site CSS (light + dark theme).
 */
(function () {
  'use strict';
  var DL = (window.DL = window.DL || {});
  DL.demos = DL.demos || {};
  var themeListeners = [];

  DL.register = function (name, fn) {
    DL.demos[name] = fn;
  };

  // ------------------------------------------------------------- DOM
  DL.el = function (tag, attrs) {
    var e = document.createElement(tag);
    attrs = attrs || {};
    Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v == null || v === false) return;
      if (k === 'class') e.className = v;
      else if (k === 'text') e.textContent = v;
      else if (k === 'html') e.innerHTML = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
      else if (k.slice(0, 2) === 'on' && typeof v === 'function') e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? '' : v);
    });
    for (var i = 2; i < arguments.length; i++) {
      var c = arguments[i];
      if (c == null) continue;
      e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    }
    return e;
  };

  DL.controls = function (root) {
    var d = DL.el('div', { class: 'demo-controls' });
    root.appendChild(d);
    return d;
  };

  // slider: opts {label,min,max,step,value,format(v)->string}; returns {el,get,set}
  DL.slider = function (parent, opts, onInput) {
    var fmt = opts.format || function (v) { return String(+(+v).toFixed(4)); };
    var out = DL.el('output', { class: 'dl-slider-val', text: fmt(opts.value) });
    var input = DL.el('input', { type: 'range', min: opts.min, max: opts.max, step: opts.step == null ? 'any' : opts.step, value: opts.value });
    var label = DL.el('label', { class: 'dl-slider' }, DL.el('span', { class: 'dl-slider-label', text: opts.label }), input, out);
    input.addEventListener('input', function () {
      out.textContent = fmt(input.value);
      if (onInput) onInput(parseFloat(input.value));
    });
    parent.appendChild(label);
    return {
      el: label,
      get: function () { return parseFloat(input.value); },
      set: function (v) { input.value = v; out.textContent = fmt(v); },
    };
  };

  // select: opts {label,options:[{value,label}],value}
  DL.select = function (parent, opts, onChange) {
    var sel = DL.el('select', {});
    opts.options.forEach(function (o) {
      var op = DL.el('option', { value: o.value, text: o.label });
      if (o.value === opts.value) op.selected = true;
      sel.appendChild(op);
    });
    var label = DL.el('label', { class: 'dl-select' }, DL.el('span', { class: 'dl-slider-label', text: opts.label }), sel);
    sel.addEventListener('change', function () { onChange && onChange(sel.value); });
    parent.appendChild(label);
    return { el: label, get: function () { return sel.value; }, set: function (v) { sel.value = v; } };
  };

  DL.button = function (parent, text, onClick, cls) {
    var b = DL.el('button', { type: 'button', class: 'dl-btn' + (cls ? ' ' + cls : ''), text: text });
    b.addEventListener('click', onClick);
    parent.appendChild(b);
    return b;
  };

  DL.toggle = function (parent, label, checked, onChange) {
    var cb = DL.el('input', { type: 'checkbox' });
    cb.checked = !!checked;
    var l = DL.el('label', { class: 'dl-toggle' }, cb, DL.el('span', { text: label }));
    cb.addEventListener('change', function () { onChange && onChange(cb.checked); });
    parent.appendChild(l);
    return { el: l, get: function () { return cb.checked; }, set: function (v) { cb.checked = !!v; } };
  };

  // readout: small monospace status box; returns function set(htmlOrText)
  DL.readout = function (parent) {
    var d = DL.el('div', { class: 'dl-readout' });
    parent.appendChild(d);
    return function (s) { d.innerHTML = s; };
  };

  DL.row = function (parent) {
    var d = DL.el('div', { class: 'dl-row' });
    parent.appendChild(d);
    return d;
  };

  // ------------------------------------------------------------- theme
  DL.colors = function () {
    var cs = getComputedStyle(document.documentElement);
    var g = function (n) { return cs.getPropertyValue(n).trim(); };
    return {
      ink: g('--c-ink'), muted: g('--c-muted'), line: g('--c-line'), bg: g('--c-bg'), soft: g('--c-soft'),
      accent: g('--c-accent'), blue: g('--c-blue'), green: g('--c-green'), purple: g('--c-purple'),
      amber: g('--c-amber'), red: g('--c-red'), teal: g('--c-teal'),
    };
  };
  DL.onTheme = function (cb) { themeListeners.push(cb); };
  DL._themeChanged = function () { themeListeners.forEach(function (f) { try { f(); } catch (e) { console.error(e); } }); };

  // ------------------------------------------------------------- canvas
  /*
   * DL.canvas(parent, {aspect: 0.55, height: 320, onResize: fn(c)}) -> c
   *   c.canvas, c.ctx (already scaled for devicePixelRatio, drawing units = CSS pixels), c.w, c.h
   *   c.redraw(): call your draw function again (set c.draw = function(){...})
   */
  DL.canvas = function (parent, opts) {
    opts = opts || {};
    var canvas = DL.el('canvas', { class: 'dl-canvas' });
    var wrap = DL.el('div', { class: 'dl-canvas-wrap' }, canvas);
    parent.appendChild(wrap);
    var c = { canvas: canvas, ctx: canvas.getContext('2d'), w: 0, h: 0, draw: null };
    function size() {
      var w = Math.max(240, wrap.clientWidth || 600);
      var h = opts.height || Math.round(w * (opts.aspect || 0.55));
      var dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.height = h + 'px';
      c.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      c.w = w;
      c.h = h;
      if (opts.onResize) opts.onResize(c);
      if (c.draw) c.draw();
    }
    c.redraw = function () { if (c.draw) c.draw(); };
    c.resize = size;
    if (window.ResizeObserver) new ResizeObserver(function () { if (wrap.clientWidth !== c._lastW) { c._lastW = wrap.clientWidth; size(); } }).observe(wrap);
    else window.addEventListener('resize', size);
    DL.onTheme(function () { c.redraw(); });
    setTimeout(size, 0);
    return c;
  };

  /*
   * DL.plot(c, {xmin,xmax,ymin,ymax, pad:{l,r,t,b}, xlabel, ylabel}) -> plot helper bound to canvas c
   *   p.clear(); p.axes(); p.grid(); p.line(xs, ys, color, width, dash); p.fn(f, color, width)
   *   p.points(xs, ys, color, radius); p.text(str, dataX, dataY, color, align); p.hline(y,color), p.vline(x,color)
   *   p.px(x), p.py(y) data->pixel ; p.dx(px), p.dy(py) pixel->data
   *   p.contour(f, levels, color) draws level sets of f(x,y)
   *   p.heat(f(x,y)->0..1, cmap) paints a heat map of the plot area
   */
  DL.plot = function (c, cfg) {
    var ctx = c.ctx;
    var P = {};
    function pad() { var p = cfg.pad || {}; return { l: p.l == null ? 44 : p.l, r: p.r == null ? 14 : p.r, t: p.t == null ? 12 : p.t, b: p.b == null ? 34 : p.b }; }
    P.cfg = cfg;
    P.px = function (x) { var p = pad(); return p.l + (x - cfg.xmin) / (cfg.xmax - cfg.xmin) * (c.w - p.l - p.r); };
    P.py = function (y) { var p = pad(); return c.h - p.b - (y - cfg.ymin) / (cfg.ymax - cfg.ymin) * (c.h - p.t - p.b); };
    P.dx = function (px) { var p = pad(); return cfg.xmin + (px - p.l) / (c.w - p.l - p.r) * (cfg.xmax - cfg.xmin); };
    P.dy = function (py) { var p = pad(); return cfg.ymin + (c.h - p.b - py) / (c.h - p.t - p.b) * (cfg.ymax - cfg.ymin); };
    P.clear = function () { ctx.clearRect(0, 0, c.w, c.h); };
    function nice(range) {
      var raw = range / 6, mag = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / mag;
      return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
    }
    P.grid = function () {
      var col = DL.colors(), p = pad();
      ctx.save();
      ctx.strokeStyle = col.line; ctx.lineWidth = 1; ctx.globalAlpha = 0.55;
      var sx = nice(cfg.xmax - cfg.xmin), sy = nice(cfg.ymax - cfg.ymin), v;
      ctx.beginPath();
      for (v = Math.ceil(cfg.xmin / sx) * sx; v <= cfg.xmax + 1e-9; v += sx) { ctx.moveTo(P.px(v), p.t); ctx.lineTo(P.px(v), c.h - p.b); }
      for (v = Math.ceil(cfg.ymin / sy) * sy; v <= cfg.ymax + 1e-9; v += sy) { ctx.moveTo(p.l, P.py(v)); ctx.lineTo(c.w - p.r, P.py(v)); }
      ctx.stroke();
      ctx.restore();
    };
    P.axes = function () {
      var col = DL.colors(), p = pad();
      ctx.save();
      ctx.strokeStyle = col.muted; ctx.fillStyle = col.muted; ctx.lineWidth = 1.2;
      ctx.font = '11px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
      ctx.beginPath(); ctx.moveTo(p.l, p.t); ctx.lineTo(p.l, c.h - p.b); ctx.lineTo(c.w - p.r, c.h - p.b); ctx.stroke();
      var sx = nice(cfg.xmax - cfg.xmin), sy = nice(cfg.ymax - cfg.ymin), v;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (v = Math.ceil(cfg.xmin / sx) * sx; v <= cfg.xmax + 1e-9; v += sx) { ctx.fillText(+v.toFixed(6) + '', P.px(v), c.h - p.b + 5); }
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (v = Math.ceil(cfg.ymin / sy) * sy; v <= cfg.ymax + 1e-9; v += sy) { ctx.fillText(+v.toFixed(6) + '', p.l - 6, P.py(v)); }
      ctx.font = '12px ' + getComputedStyle(document.body).getPropertyValue('--font-sans');
      if (cfg.xlabel) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(cfg.xlabel, (p.l + c.w - p.r) / 2, c.h - 1); }
      if (cfg.ylabel) { ctx.save(); ctx.translate(11, (p.t + c.h - p.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(cfg.ylabel, 0, 0); ctx.restore(); }
      ctx.restore();
    };
    P.line = function (xs, ys, color, width, dash) {
      var p = pad();
      ctx.save();
      ctx.beginPath(); ctx.rect(p.l, p.t, c.w - p.l - p.r, c.h - p.t - p.b); ctx.clip();
      ctx.strokeStyle = color || DL.colors().accent; ctx.lineWidth = width || 2; ctx.lineJoin = 'round';
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      var started = false;
      for (var i = 0; i < xs.length; i++) {
        if (!isFinite(ys[i])) { started = false; continue; }
        var X = P.px(xs[i]), Y = P.py(ys[i]);
        if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
      }
      ctx.stroke();
      ctx.restore();
    };
    P.fn = function (f, color, width, dash, n) {
      n = n || 240;
      var xs = [], ys = [];
      for (var i = 0; i <= n; i++) { var x = cfg.xmin + (cfg.xmax - cfg.xmin) * i / n; xs.push(x); ys.push(f(x)); }
      P.line(xs, ys, color, width, dash);
    };
    P.points = function (xs, ys, color, r) {
      ctx.save(); ctx.fillStyle = color || DL.colors().blue;
      for (var i = 0; i < xs.length; i++) { ctx.beginPath(); ctx.arc(P.px(xs[i]), P.py(ys[i]), r || 3.2, 0, Math.PI * 2); ctx.fill(); }
      ctx.restore();
    };
    P.hline = function (y, color, dash) { P.line([cfg.xmin, cfg.xmax], [y, y], color || DL.colors().muted, 1.2, dash || [5, 4]); };
    P.vline = function (x, color, dash) { P.line([x, x], [cfg.ymin, cfg.ymax], color || DL.colors().muted, 1.2, dash || [5, 4]); };
    P.text = function (str, x, y, color, align) {
      ctx.save(); ctx.fillStyle = color || DL.colors().ink; ctx.textAlign = align || 'left'; ctx.textBaseline = 'middle';
      ctx.font = '12px ' + getComputedStyle(document.body).getPropertyValue('--font-sans');
      ctx.fillText(str, P.px(x), P.py(y)); ctx.restore();
    };
    P.heat = function (f, cmap, res) {
      var p = pad(), cw = c.w - p.l - p.r, ch = c.h - p.t - p.b;
      res = res || 4;
      cmap = cmap || DL.cmap.viridis;
      for (var i = 0; i < cw; i += res) for (var j = 0; j < ch; j += res) {
        var v = f(P.dx(p.l + i + res / 2), P.dy(p.t + j + res / 2));
        ctx.fillStyle = cmap(Math.max(0, Math.min(1, v)));
        ctx.fillRect(p.l + i, p.t + j, res, res);
      }
    };
    P.contour = function (f, levels, color) {
      var p = pad(), step = 6, cw = c.w - p.l - p.r, ch = c.h - p.t - p.b;
      var nx = Math.ceil(cw / step), ny = Math.ceil(ch / step);
      var g = [];
      for (var i = 0; i <= nx; i++) { g[i] = []; for (var j = 0; j <= ny; j++) g[i][j] = f(P.dx(p.l + i * step), P.dy(p.t + j * step)); }
      ctx.save(); ctx.beginPath(); ctx.rect(p.l, p.t, cw, ch); ctx.clip();
      ctx.strokeStyle = color || DL.colors().muted; ctx.lineWidth = 1; ctx.globalAlpha = 0.8;
      levels.forEach(function (L) {
        ctx.beginPath();
        for (var i = 0; i < nx; i++) for (var j = 0; j < ny; j++) {
          var a = g[i][j] - L, b = g[i + 1][j] - L, cc = g[i + 1][j + 1] - L, d = g[i][j + 1] - L;
          var x0 = p.l + i * step, y0 = p.t + j * step, pts = [];
          function lerp(v0, v1) { return v0 / (v0 - v1); }
          if (a * b < 0) pts.push([x0 + lerp(a, b) * step, y0]);
          if (b * cc < 0) pts.push([x0 + step, y0 + lerp(b, cc) * step]);
          if (d * cc < 0) pts.push([x0 + lerp(d, cc) * step, y0 + step]);
          if (a * d < 0) pts.push([x0, y0 + lerp(a, d) * step]);
          if (pts.length >= 2) { ctx.moveTo(pts[0][0], pts[0][1]); ctx.lineTo(pts[1][0], pts[1][1]); }
          if (pts.length === 4) { ctx.moveTo(pts[2][0], pts[2][1]); ctx.lineTo(pts[3][0], pts[3][1]); }
        }
        ctx.stroke();
      });
      ctx.restore();
    };
    return P;
  };

  // ------------------------------------------------------------- color maps (t in 0..1 -> css color)
  function lerpColor(stops, t) {
    t = Math.max(0, Math.min(1, t));
    var n = stops.length - 1, i = Math.min(n - 1, Math.floor(t * n)), f = t * n - i, a = stops[i], b = stops[i + 1];
    return 'rgb(' + Math.round(a[0] + (b[0] - a[0]) * f) + ',' + Math.round(a[1] + (b[1] - a[1]) * f) + ',' + Math.round(a[2] + (b[2] - a[2]) * f) + ')';
  }
  DL.cmap = {
    viridis: function (t) { return lerpColor([[68, 1, 84], [59, 82, 139], [33, 144, 141], [93, 201, 99], [253, 231, 37]], t); },
    magma: function (t) { return lerpColor([[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]], t); },
    // diverging blue-white-red for signed values: pass t = 0.5 + v/2 for v in [-1,1]
    diverge: function (t) { return lerpColor([[33, 102, 172], [146, 197, 222], [247, 247, 247], [244, 165, 130], [178, 24, 43]], t); },
    gray: function (t) { var g = Math.round(255 * Math.max(0, Math.min(1, t))); return 'rgb(' + g + ',' + g + ',' + g + ')'; },
    warm: function (t) { return lerpColor([[250, 246, 238], [244, 214, 170], [217, 119, 87], [120, 50, 40]], t); },
  };

  /*
   * DL.heatmap(ctx, matrix(2D array of numbers), x, y, w, h, {min,max,cmap,labels:bool,rowLabels,colLabels,border})
   * Draws a matrix as colored cells (optionally printing the values).
   */
  DL.heatmap = function (ctx, M, x, y, w, h, o) {
    o = o || {};
    var R = M.length, C = M[0].length, cw = w / C, ch = h / R;
    var mn = o.min, mx = o.max, i, j;
    if (mn == null || mx == null) { mn = Infinity; mx = -Infinity; for (i = 0; i < R; i++) for (j = 0; j < C; j++) { mn = Math.min(mn, M[i][j]); mx = Math.max(mx, M[i][j]); } if (mx === mn) mx = mn + 1; }
    var cmap = o.cmap || DL.cmap.viridis, col = DL.colors();
    ctx.save();
    for (i = 0; i < R; i++) for (j = 0; j < C; j++) {
      var t = (M[i][j] - mn) / (mx - mn);
      ctx.fillStyle = cmap(t); ctx.fillRect(x + j * cw, y + i * ch, cw + 0.5, ch + 0.5);
      if (o.labels && cw > 22 && ch > 14) {
        ctx.fillStyle = t > 0.55 && cmap === DL.cmap.viridis ? '#111' : (t > 0.5 ? '#fff' : col.ink);
        if (cmap === DL.cmap.gray) ctx.fillStyle = t > 0.5 ? '#111' : '#eee';
        ctx.font = Math.min(12, ch * 0.5) + 'px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(o.fmt ? o.fmt(M[i][j]) : (+M[i][j].toFixed(2)) + '', x + (j + 0.5) * cw, y + (i + 0.5) * ch);
      }
    }
    if (o.border !== false) { ctx.strokeStyle = col.line; ctx.lineWidth = 1; ctx.strokeRect(x, y, w, h); }
    ctx.fillStyle = col.muted; ctx.font = '11px ' + getComputedStyle(document.body).getPropertyValue('--font-sans');
    if (o.colLabels) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; o.colLabels.forEach(function (s, k) { ctx.fillText(s, x + (k + 0.5) * cw, y - 3); }); }
    if (o.rowLabels) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; o.rowLabels.forEach(function (s, k) { ctx.fillText(s, x - 4, y + (k + 0.5) * ch); }); }
    ctx.restore();
  };

  // ------------------------------------------------------------- math / random
  DL.rng = function (seed) {
    var s = seed >>> 0 || 123456789;
    return function () { s += 0x6D2B79F5; var t = Math.imul(s ^ (s >>> 15), 1 | s); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  };
  DL.randn = function (rand) {
    rand = rand || Math.random;
    var u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  DL.sum = function (a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; };
  DL.softmax = function (z, T) {
    T = T || 1;
    var m = Math.max.apply(null, z), e = z.map(function (v) { return Math.exp((v - m) / T); }), s = DL.sum(e);
    return e.map(function (v) { return v / s; });
  };
  DL.sigmoid = function (x) { return 1 / (1 + Math.exp(-x)); };
  DL.matmul = function (A, B) {
    var n = A.length, m = B[0].length, k = B.length, C = [], i, j, t;
    for (i = 0; i < n; i++) { C[i] = []; for (j = 0; j < m; j++) { var s = 0; for (t = 0; t < k; t++) s += A[i][t] * B[t][j]; C[i][j] = s; } }
    return C;
  };
  DL.transpose = function (A) { return A[0].map(function (_, j) { return A.map(function (r) { return r[j]; }); }); };
  DL.clamp = function (v, a, b) { return Math.max(a, Math.min(b, v)); };
  DL.fmt = function (v, d) { return (+v).toFixed(d == null ? 3 : d); };
  DL.range = function (n) { var a = []; for (var i = 0; i < n; i++) a.push(i); return a; };
  DL.linspace = function (a, b, n) { var r = []; for (var i = 0; i < n; i++) r.push(a + (b - a) * i / (n - 1)); return r; };

  // drag helper: DL.drag(canvas, {down(x,y), move(x,y), up()}) with CSS-pixel coordinates
  DL.drag = function (canvas, h) {
    function pos(e) { var r = canvas.getBoundingClientRect(); var t = e.touches ? e.touches[0] : e; return [t.clientX - r.left, t.clientY - r.top]; }
    var down = false;
    function md(e) { down = true; var p = pos(e); h.down && h.down(p[0], p[1]); e.preventDefault(); }
    function mm(e) { var p = pos(e); if (down) { h.move && h.move(p[0], p[1]); e.preventDefault(); } else h.hover && h.hover(p[0], p[1]); }
    function mu() { if (down) { down = false; h.up && h.up(); } }
    canvas.addEventListener('mousedown', md); canvas.addEventListener('touchstart', md, { passive: false });
    window.addEventListener('mousemove', mm); canvas.addEventListener('touchmove', mm, { passive: false });
    window.addEventListener('mouseup', mu); window.addEventListener('touchend', mu);
    canvas.style.touchAction = 'none';
  };
})();
