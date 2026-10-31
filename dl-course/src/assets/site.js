/* Site behaviour: theme, nav, TOC spy, search, progress, copy buttons, charts, lazy demos. No dependencies. */
(function () {
  'use strict';
  var doc = document, root = doc.documentElement, body = doc.body;
  var $ = function (s, r) { return (r || doc).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || doc).querySelectorAll(s)); };
  var store = {
    get: function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
  };

  /* ------------------------------------------------ theme */
  function setTheme(t) {
    root.setAttribute('data-theme', t);
    store.set('dl-theme', t);
    if (window.DL && DL._themeChanged) DL._themeChanged();
    refreshCharts();
  }
  var themeBtn = $('#theme-btn');
  if (themeBtn) themeBtn.addEventListener('click', function () {
    setTheme(root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  });

  /* ------------------------------------------------ mobile nav */
  var menuBtn = $('#menu-btn'), scrim = $('#scrim');
  function navOpen(o) {
    body.classList.toggle('nav-open', o);
    if (scrim) scrim.hidden = !o;
    if (menuBtn) menuBtn.setAttribute('aria-expanded', String(o));
  }
  if (menuBtn) menuBtn.addEventListener('click', function () { navOpen(!body.classList.contains('nav-open')); });
  if (scrim) scrim.addEventListener('click', function () { navOpen(false); });
  var active = $('.side a.active');
  if (active && active.scrollIntoView) { var sb = $('#sidebar'); if (sb) sb.scrollTop = Math.max(0, active.offsetTop - sb.clientHeight / 3); }

  /* ------------------------------------------------ progress (localStorage) */
  function getDone() { try { return JSON.parse(store.get('dl-done') || '{}'); } catch (e) { return {}; } }
  function paintDone() {
    var d = getDone();
    $$('[data-week]').forEach(function (el) {
      if (el.tagName === 'A' || el.classList.contains('week-card')) {
        var li = el.closest('li') || el;
        if (d[el.getAttribute('data-week')]) li.setAttribute('data-done', '1'); else li.removeAttribute('data-done');
        if (el.classList.contains('week-card')) { var dn = $('.done', el); if (dn) dn.classList.toggle('on', !!d[el.getAttribute('data-week')]); }
      }
    });
    var btn = $('#mark-complete');
    if (btn) btn.classList.toggle('on', !!d[btn.getAttribute('data-week')]);
    var cont = $('#continue-btn');
    if (cont) {
      var next = null, last = 0;
      $$('.week-card').forEach(function (c) { var n = c.getAttribute('data-week'); if (d[n]) last = Math.max(last, +n); });
      var target = $('.week-card[data-week="' + (last + 1) + '"]');
      if (last > 0 && target) { cont.textContent = 'Continue with Week ' + (last + 1); cont.setAttribute('href', target.getAttribute('href')); }
      else if (last > 0) cont.textContent = 'Review the course';
    }
  }
  var markBtn = $('#mark-complete');
  if (markBtn) markBtn.addEventListener('click', function () {
    var d = getDone(), w = markBtn.getAttribute('data-week');
    if (d[w]) delete d[w]; else d[w] = 1;
    store.set('dl-done', JSON.stringify(d)); paintDone();
  });
  var reset = $('#reset-progress');
  if (reset) reset.addEventListener('click', function () {
    if (confirm('Clear your saved progress in this browser?')) { store.set('dl-done', '{}'); paintDone(); }
  });
  paintDone();

  /* ------------------------------------------------ reading progress + TOC scroll-spy */
  var bar = $('#read-progress'), prose = $('.prose');
  var heads = $$('.prose h2, .prose h3');
  var tocLinks = {};
  $$('.toc a').forEach(function (a) { tocLinks[a.getAttribute('href').slice(1)] = a; });
  var ticking = false;
  function onScroll() {
    ticking = false;
    if (bar && prose) {
      var r = prose.getBoundingClientRect(), total = r.height - window.innerHeight * 0.6;
      bar.style.width = Math.max(0, Math.min(1, -r.top / Math.max(1, total))) * 100 + '%';
    }
    if (heads.length && Object.keys(tocLinks).length) {
      var cur = null, y = 90;
      for (var i = 0; i < heads.length; i++) { if (heads[i].getBoundingClientRect().top - y <= 0) cur = heads[i]; else break; }
      Object.keys(tocLinks).forEach(function (k) { tocLinks[k].classList.remove('active'); });
      if (cur && tocLinks[cur.id]) {
        tocLinks[cur.id].classList.add('active');
        var t = $('.toc'); var a = tocLinks[cur.id];
        if (t && (a.offsetTop < t.scrollTop + 40 || a.offsetTop > t.scrollTop + t.clientHeight - 60)) t.scrollTop = a.offsetTop - t.clientHeight / 2;
      }
    }
  }
  window.addEventListener('scroll', function () { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });
  onScroll();

  /* ------------------------------------------------ copy buttons */
  $$('.copy-btn').forEach(function (b) {
    b.addEventListener('click', function () {
      var code = b.closest('.codeblock').querySelector('code').innerText;
      var done = function () { b.textContent = 'Copied'; setTimeout(function () { b.textContent = 'Copy'; }, 1400); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(code).then(done, fallback);
      else fallback();
      function fallback() {
        var ta = doc.createElement('textarea'); ta.value = code; ta.style.position = 'fixed'; ta.style.opacity = '0';
        body.appendChild(ta); ta.select(); try { doc.execCommand('copy'); done(); } catch (e) {} body.removeChild(ta);
      }
    });
  });

  /* ------------------------------------------------ charts (Chart.js) */
  var charts = [];
  var palette = ['--c-blue', '--c-accent', '--c-green', '--c-purple', '--c-amber', '--c-teal', '--c-red'];
  function cssVar(n) { return getComputedStyle(root).getPropertyValue(n).trim(); }
  function resolve(v) {
    if (typeof v === 'string') { var m = v.match(/^var\((--[\w-]+)\)$/); if (m) return cssVar(m[1]); }
    return v;
  }
  function deepResolve(o) {
    if (Array.isArray(o)) return o.map(deepResolve);
    if (o && typeof o === 'object') { var r = {}; Object.keys(o).forEach(function (k) { r[k] = deepResolve(o[k]); }); return r; }
    return resolve(o);
  }
  function tint(hex, a) {
    var m = /^#?([0-9a-f]{6})$/i.exec(hex || ''); if (!m) return hex;
    var n = parseInt(m[1], 16); return 'rgba(' + (n >> 16) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
  }
  function buildChart(canvas) {
    if (!window.Chart) return;
    var raw; try { raw = JSON.parse(canvas.getAttribute('data-chart')); } catch (e) { return; }
    var spec = deepResolve(raw), ink = cssVar('--c-ink2'), grid = cssVar('--c-line'), mut = cssVar('--c-muted');
    var type = spec.type, pieLike = type === 'doughnut' || type === 'pie';
    spec.data = spec.data || {}; spec.data.datasets = spec.data.datasets || [];
    spec.data.datasets.forEach(function (ds, i) {
      var c = cssVar(palette[i % palette.length]);
      if (pieLike) { if (!ds.backgroundColor) ds.backgroundColor = (spec.data.labels || ds.data).map(function (_, j) { return cssVar(palette[j % palette.length]); }); ds.borderColor = ds.borderColor || cssVar('--c-surface'); return; }
      if (!ds.borderColor) ds.borderColor = c;
      if (!ds.backgroundColor) ds.backgroundColor = (type === 'bar') ? tint(c, 0.75) : (type === 'line' && ds.fill ? tint(c, 0.15) : c);
      if (type === 'line') { if (ds.borderWidth == null) ds.borderWidth = 2.2; if (ds.pointRadius == null) ds.pointRadius = (ds.data && ds.data.length > 40) ? 0 : 3; if (ds.tension == null) ds.tension = 0; }
      if (type === 'scatter' && ds.pointRadius == null) ds.pointRadius = 3.5;
    });
    var o = spec.options = spec.options || {};
    o.responsive = true; o.maintainAspectRatio = false; o.animation = o.animation || { duration: 350 };
    var plugins = o.plugins = o.plugins || {};
    plugins.legend = plugins.legend || {};
    if (plugins.legend.display == null) plugins.legend.display = spec.data.datasets.length > 1 || pieLike;
    plugins.legend.labels = Object.assign({ color: ink, usePointStyle: true, boxWidth: 8, font: { family: cssVar('--font-sans'), size: 12 } }, plugins.legend.labels || {});
    plugins.tooltip = Object.assign({ backgroundColor: cssVar('--c-ink'), titleColor: cssVar('--c-bg'), bodyColor: cssVar('--c-bg'), padding: 10, cornerRadius: 8 }, plugins.tooltip || {});
    if (!pieLike && type !== 'radar') {
      var sc = o.scales = o.scales || {};
      ['x', 'y'].forEach(function (a) {
        var s = sc[a] = sc[a] || {};
        s.grid = Object.assign({ color: grid }, s.grid || {});
        s.ticks = Object.assign({ color: mut, font: { family: cssVar('--font-sans'), size: 11.5 } }, s.ticks || {});
        s.border = Object.assign({ color: cssVar('--c-line2') }, s.border || {});
        if (s.title) s.title = Object.assign({ color: mut, font: { family: cssVar('--font-sans'), size: 12, weight: '600' } }, s.title);
      });
    } else if (type === 'radar') {
      o.scales = o.scales || { r: {} };
      o.scales.r = Object.assign({ grid: { color: grid }, angleLines: { color: grid }, pointLabels: { color: ink }, ticks: { color: mut, backdropColor: 'transparent' } }, o.scales.r || {});
    }
    var ch = new Chart(canvas.getContext('2d'), spec);
    charts.push({ canvas: canvas, chart: ch });
  }
  function refreshCharts() {
    charts.forEach(function (c) { try { c.chart.destroy(); } catch (e) {} });
    charts = [];
    $$('canvas[data-chart]').forEach(buildChart);
  }
  function initCharts() {
    var cs = $$('canvas[data-chart]'); if (!cs.length) return;
    if (!window.Chart) { setTimeout(initCharts, 60); return; }
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (es) {
        es.forEach(function (e) { if (e.isIntersecting) { io.unobserve(e.target); buildChart(e.target); } });
      }, { rootMargin: '300px' });
      cs.forEach(function (c) { io.observe(c); });
    } else cs.forEach(buildChart);
  }

  /* ------------------------------------------------ lazy demos */
  function startDemo(el) {
    var name = el.getAttribute('data-demo'), body_ = $('.demo-body', el);
    if (el.getAttribute('data-started')) return;
    if (!window.DL || !DL.demos || !DL.demos[name]) {
      if ((el._tries = (el._tries || 0) + 1) < 40) { setTimeout(function () { startDemo(el); }, 80); return; }
      body_.innerHTML = '<div class="demo-error">Demo "' + name + '" failed to load.</div>'; return;
    }
    el.setAttribute('data-started', '1'); body_.innerHTML = '';
    try { DL.demos[name](body_); }
    catch (e) { console.error(e); body_.innerHTML = '<div class="demo-error">This demo hit an error: ' + String(e.message || e).replace(/</g, '&lt;') + '</div>'; }
  }
  function initDemos() {
    var ds = $$('.demo[data-demo]'); if (!ds.length) return;
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (es) {
        es.forEach(function (e) { if (e.isIntersecting) { io.unobserve(e.target); startDemo(e.target); } });
      }, { rootMargin: '400px' });
      ds.forEach(function (d) { io.observe(d); });
    } else ds.forEach(startDemo);
  }

  /* ------------------------------------------------ search */
  var modal = $('#search-modal'), input = $('#search-input'), list = $('#search-results'), sel = -1, results = [];
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function norm(s) { return String(s).toLowerCase(); }
  function runSearch(q) {
    var idx = window.DL_SEARCH || []; q = norm(q).trim();
    if (!q) { results = []; render(); return; }
    var terms = q.split(/\s+/);
    var scored = [];
    idx.forEach(function (d) {
      var h = norm(d.h), t = norm(d.t), x = norm(d.x), s = 0, ok = true;
      terms.forEach(function (w) {
        var sc = 0;
        if (h.indexOf(w) >= 0) sc += 12 + (h.indexOf(w) === 0 ? 4 : 0);
        if (t.indexOf(w) >= 0) sc += 5;
        var c = 0, p = -1; while ((p = x.indexOf(w, p + 1)) >= 0 && c < 6) c++;
        sc += c * 2;
        if (!sc) ok = false; s += sc;
      });
      if (ok) scored.push({ d: d, s: s });
    });
    scored.sort(function (a, b) { return b.s - a.s; });
    results = scored.slice(0, 9).map(function (r) { return r.d; });
    render(terms);
  }
  function snippet(d, terms) {
    var x = d.x, lx = norm(x), pos = -1;
    for (var i = 0; i < terms.length && pos < 0; i++) pos = lx.indexOf(terms[i]);
    var a = Math.max(0, (pos < 0 ? 0 : pos) - 50), s = x.slice(a, a + 160);
    s = esc(s);
    terms.forEach(function (w) { if (w.length > 1) s = s.replace(new RegExp('(' + w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig'), '<mark>$1</mark>'); });
    return (a > 0 ? '…' : '') + s + '…';
  }
  function render(terms) {
    sel = results.length ? 0 : -1;
    if (!results.length) { list.innerHTML = input.value.trim() ? '<li class="sr-empty">No results. Try fewer or different words.</li>' : '<li class="sr-empty">Search across every week: try “backpropagation”, “attention”, “KL divergence”…</li>'; return; }
    list.innerHTML = results.map(function (d, i) {
      return '<li' + (i === 0 ? ' class="sel"' : '') + '><a href="' + d.p + (d.a ? '#' + d.a : '') + '"><div class="sr-top"><span>' + esc(d.w) + '</span><span>' + esc(d.t) + '</span></div><div class="sr-h">' + esc(d.h) + '</div><div class="sr-x">' + snippet(d, terms || []) + '</div></a></li>';
    }).join('');
  }
  function openSearch() { if (!modal) return; modal.hidden = false; input.value = ''; render(); input.focus(); }
  function closeSearch() { if (modal) modal.hidden = true; }
  function move(dx) {
    var items = $$('li', list); if (!items.length || sel < 0) return;
    items[sel].classList.remove('sel'); sel = (sel + dx + items.length) % items.length; items[sel].classList.add('sel');
    items[sel].scrollIntoView({ block: 'nearest' });
  }
  var sbtn = $('#search-btn'); if (sbtn) sbtn.addEventListener('click', openSearch);
  if (modal) {
    modal.addEventListener('mousedown', function (e) { if (e.target === modal) closeSearch(); });
    input.addEventListener('input', function () { runSearch(input.value); });
    input.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown') { e.preventDefault(); move(1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); move(-1); }
      else if (e.key === 'Enter') { var a = $('li.sel a', list); if (a) location.href = a.getAttribute('href'); }
    });
  }
  doc.addEventListener('keydown', function (e) {
    var tag = (e.target.tagName || '').toLowerCase(), typing = tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable;
    if ((e.key === '/' && !typing) || ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k')) { e.preventDefault(); openSearch(); }
    else if (e.key === 'Escape') { closeSearch(); navOpen(false); }
  });

  /* ------------------------------------------------ boot */
  function boot() { initCharts(); initDemos(); }
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', boot); else boot();
})();
