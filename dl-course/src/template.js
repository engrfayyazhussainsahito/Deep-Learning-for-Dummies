'use strict';
const esc = (s) => String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const pad2 = (n) => String(n).padStart(2, '0');

const SITE = {
  name: 'Deep Learning',
  tagline: 'An open, graduate-level course on deep learning, from perceptrons to diffusion models.',
  repo: 'https://github.com/your-name/deep-learning-course',
};

const themeBoot = `(function(){try{var t=localStorage.getItem('dl-theme');if(!t){t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'}document.documentElement.setAttribute('data-theme',t)}catch(e){}})();`;

function head({ title, desc, hasChart, hasMath }) {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<meta name="description" content="${esc(desc || SITE.tagline)}">
<meta name="theme-color" content="#faf7f2">
<meta property="og:title" content="${esc(title)}">
<meta property="og:description" content="${esc(desc || SITE.tagline)}">
<meta property="og:type" content="website">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
<script>${themeBoot}</script>
<link rel="stylesheet" href="assets/vendor/katex/katex.min.css">
<link rel="stylesheet" href="assets/site.css">
</head>`;
}

function sidebar(nav, activeSlug) {
  const byWeek = new Map(nav.weeks.map((w) => [w.n, w]));
  let h = `<nav class="side" id="sidebar" aria-label="Course navigation"><div class="side-inner">`;
  h += `<ul class="side-top"><li><a href="index.html"${activeSlug === 'index' ? ' class="active"' : ''}>Home</a></li>`;
  h += `<li><a href="syllabus.html"${activeSlug === 'syllabus' ? ' class="active"' : ''}>Syllabus</a></li>`;
  h += `<li><a href="glossary.html"${activeSlug === 'glossary' ? ' class="active"' : ''}>Glossary</a></li></ul>`;
  for (const ph of nav.phases) {
    h += `<div class="side-phase"><div class="side-phase-title"><span class="phase-num">${ph.id}</span>${esc(ph.name)}</div><ul>`;
    for (const n of ph.weeks) {
      const w = byWeek.get(n);
      if (!w) {
        h += `<li class="soon"><span class="wk">${pad2(n)}</span><span class="wt">Week ${n}</span><em>soon</em></li>`;
        continue;
      }
      h += `<li><a href="${w.slug}.html" data-week="${w.n}"${activeSlug === w.slug ? ' class="active" aria-current="page"' : ''}><span class="wk">${pad2(w.n)}</span><span class="wt">${esc(w.title)}</span><span class="done" aria-hidden="true"></span></a></li>`;
    }
    h += `</ul></div>`;
  }
  h += `<div class="side-foot"><a href="about.html">About &amp; contributing</a><button type="button" class="linkbtn" id="reset-progress">Reset progress</button></div>`;
  h += `</div></nav>`;
  return h;
}

function topbar() {
  return `<header class="top">
<button class="icon-btn menu-btn" id="menu-btn" type="button" aria-label="Open navigation" aria-expanded="false" aria-controls="sidebar"><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 6h16M4 12h16M4 18h16"/></svg></button>
<a class="brand" href="index.html"><svg viewBox="0 0 32 32" width="24" height="24" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="6" cy="9" r="3"/><circle cx="6" cy="23" r="3"/><circle cx="16" cy="16" r="3"/><circle cx="26" cy="9" r="3" fill="currentColor"/><circle cx="26" cy="23" r="3"/><path d="M9 10l4.5 4.2M9 22l4.5-4.2M19 14.5L23 10.5M19 17.5L23 21.5" stroke-width="1.5"/></g></svg><span>${SITE.name}</span></a>
<div class="top-spacer"></div>
<button class="search-btn" id="search-btn" type="button" aria-label="Search the course"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg><span>Search</span><kbd>/</kbd></button>
<button class="icon-btn" id="theme-btn" type="button" aria-label="Toggle dark mode"><svg class="i-sun" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg><svg class="i-moon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z"/></svg></button>
</header>
<div class="progress-bar" id="read-progress" aria-hidden="true"></div>`;
}

const searchDialog = `<div class="search-modal" id="search-modal" hidden role="dialog" aria-modal="true" aria-label="Search">
<div class="search-box"><input id="search-input" type="search" placeholder="Search topics, formulas, papers..." autocomplete="off" spellcheck="false" aria-label="Search query"><ul id="search-results" role="listbox"></ul><div class="search-hint"><span><kbd>↑</kbd><kbd>↓</kbd> navigate</span><span><kbd>Enter</kbd> open</span><span><kbd>Esc</kbd> close</span></div></div></div>`;

function scripts({ hasChart, demos }) {
  let s = `<script src="assets/search-index.js" defer></script>\n`;
  if (hasChart) s += `<script src="assets/vendor/chart.umd.js" defer></script>\n`;
  s += `<script src="assets/dl-lib.js" defer></script>\n`;
  for (const d of demos || []) s += `<script src="assets/demos/${esc(d)}.js" defer></script>\n`;
  s += `<script src="assets/site.js" defer></script>`;
  return s;
}

function footer() {
  return `<footer class="site-foot"><div>Open educational resource. Text and figures licensed <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener">CC BY 4.0</a>, code under the MIT license.</div><div><a href="about.html">About &amp; contributing</a> · <a href="syllabus.html">Syllabus</a> · <a href="glossary.html">Glossary</a></div></footer>`;
}

function shell({ nav, slug, title, desc, body, toc, hasChart, demos, tocTitle }) {
  const tocHtml =
    toc && toc.length
      ? `<aside class="toc" aria-label="On this page"><div class="toc-inner"><div class="toc-title">${tocTitle || 'On this page'}</div><ul>${toc
          .map((t) => `<li class="l${t.level}"><a href="#${esc(t.id)}">${esc(t.text)}</a></li>`)
          .join('')}</ul></div></aside>`
      : '';
  return `${head({ title, desc, hasChart })}
<body data-page="${esc(slug)}">
<a class="skip" href="#main">Skip to content</a>
${topbar()}
<div class="layout">
${sidebar(nav, slug)}
<div class="scrim" id="scrim" hidden></div>
<main id="main" class="main">${body}${footer()}</main>
${tocHtml}
</div>
${searchDialog}
${scripts({ hasChart, demos })}
</body>
</html>
`;
}

// ---------------------------------------------------------------------------

function week({ nav, week: w, prev, next, html, toc, minutes, stats, words, demos, hasChart, phase }) {
  const chips = [];
  chips.push(`<span class="chip">${minutes} min read</span>`);
  if (stats.demos) chips.push(`<span class="chip">${stats.demos} interactive demo${stats.demos > 1 ? 's' : ''}</span>`);
  if (stats.figures + stats.charts) chips.push(`<span class="chip">${stats.figures + stats.charts} figures</span>`);
  if (stats.code) chips.push(`<span class="chip">${stats.code} code blocks</span>`);
  if (stats.exercises) chips.push(`<span class="chip">${stats.exercises} exercises</span>`);
  const pager = `<nav class="pager" aria-label="Previous and next week">${
    prev ? `<a class="pager-prev" href="${prev.slug}.html"><span class="pager-dir">Previous</span><span class="pager-title">Week ${prev.n}: ${esc(prev.title)}</span></a>` : '<span></span>'
  }${next ? `<a class="pager-next" href="${next.slug}.html"><span class="pager-dir">Next</span><span class="pager-title">Week ${next.n}: ${esc(next.title)}</span></a>` : '<span></span>'}</nav>`;
  const body = `<article class="article">
<header class="article-head">
<div class="eyebrow">Week ${w.n} <span class="sep">·</span> ${phase ? `Phase ${phase.id}: ${esc(phase.name)}` : esc(w.phase)}</div>
<h1>${esc(w.title)}</h1>
<p class="lede">${esc(w.summary)}</p>
<div class="chips">${chips.join('')}</div>
</header>
<div class="prose" data-week="${w.n}">${html}</div>
<div class="complete-row"><button type="button" class="btn-complete" id="mark-complete" data-week="${w.n}"><span class="bc-off">Mark week ${w.n} as complete</span><span class="bc-on">Completed. Nice work!</span></button></div>
${pager}
</article>`;
  return shell({ nav, slug: w.slug, title: `Week ${w.n}: ${w.title} | ${SITE.name}`, desc: w.summary, body, toc, hasChart, demos });
}

function doc({ nav, slug, meta, html, toc, hasChart, demos }) {
  const body = `<article class="article">
<header class="article-head"><div class="eyebrow">${esc(meta.eyebrow || 'Course')}</div><h1>${esc(meta.title || slug)}</h1>${meta.summary ? `<p class="lede">${esc(meta.summary)}</p>` : ''}</header>
<div class="prose">${html}</div></article>`;
  return shell({ nav, slug, title: `${meta.title || slug} | ${SITE.name}`, desc: meta.summary, body, toc: (toc || []).filter((t) => t.level === 2), hasChart, demos });
}

function home({ nav, weeks, totalWords, totals }) {
  totals = totals || { figures: 0, charts: 0, demos: 0, exercises: 0, code: 0 };
  const byWeek = new Map(weeks.map((w) => [w.n, w]));
  const phaseSvg = [
    // tiny abstract glyph per phase
    '<svg viewBox="0 0 64 40" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="10" cy="8" r="3.5"/><circle cx="10" cy="20" r="3.5"/><circle cx="10" cy="32" r="3.5"/><circle cx="32" cy="14" r="3.5"/><circle cx="32" cy="26" r="3.5"/><circle cx="54" cy="20" r="3.5"/><path d="M13.5 8L28.5 14M13.5 8L28.5 26M13.5 20L28.5 14M13.5 20L28.5 26M13.5 32L28.5 14M13.5 32L28.5 26M35.5 14L50.5 20M35.5 26L50.5 20" stroke-width="1"/></g></svg>',
    '<svg viewBox="0 0 64 40" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="1.6"><rect x="4" y="6" width="16" height="16" rx="2"/><rect x="12" y="12" width="16" height="16" rx="2"/><rect x="20" y="18" width="16" height="16" rx="2"/><path d="M42 20h14M52 15l5 5-5 5"/></g></svg>',
    '<svg viewBox="0 0 64 40" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="20" r="4"/><circle cx="32" cy="8" r="4"/><circle cx="32" cy="32" r="4"/><circle cx="52" cy="20" r="4"/><path d="M16 18L28 10M16 22L28 30M36 10L48 18M36 30L48 22M32 12v16" stroke-width="1.2"/></g></svg>',
    '<svg viewBox="0 0 64 40" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 30C14 30 14 8 24 8s10 22 20 22 8-14 16-14"/><circle cx="24" cy="8" r="2.5" fill="currentColor"/><circle cx="44" cy="30" r="2.5" fill="currentColor"/></g></svg>',
  ];
  const phaseCards = nav.phases
    .map((ph, i) => {
      const items = ph.weeks
        .map((n) => {
          const w = byWeek.get(n);
          return w
            ? `<li><a href="${w.slug}.html" data-week="${n}"><span class="wk">${pad2(n)}</span><span class="wt">${esc(w.title)}</span><span class="done" aria-hidden="true"></span></a></li>`
            : `<li class="soon"><span class="wk">${pad2(n)}</span><span class="wt">Week ${n}</span><em>coming soon</em></li>`;
        })
        .join('');
      return `<section class="phase-card"><div class="phase-glyph">${phaseSvg[i]}</div><div class="phase-meta">Phase ${ph.id} · Weeks ${ph.weeks[0]} to ${ph.weeks[ph.weeks.length - 1]}</div><h3>${esc(ph.name)}</h3><p>${esc(ph.blurb)}</p><ul class="phase-list">${items}</ul></section>`;
    })
    .join('');

  const cards = weeks
    .map(
      (w) =>
        `<a class="week-card" href="${w.slug}.html" data-week="${w.n}"><div class="wc-top"><span class="wc-num">${pad2(w.n)}</span><span class="done" aria-hidden="true"></span></div><h4>${esc(w.title)}</h4><p>${esc(w.summary)}</p></a>`
    )
    .join('');

  const body = `<div class="home">
<section class="hero">
<div class="hero-copy">
<div class="eyebrow">Open course · MS level · 18 weeks</div>
<h1>Learn deep learning<br>the way it is <em>actually</em> done.</h1>
<p class="hero-sub">${SITE.tagline} Every week pairs rigorous derivations with runnable PyTorch, original diagrams and interactive experiments you can poke at in your browser. Free, open source and built to be studied from end to end.</p>
<div class="hero-cta"><a class="btn btn-primary" id="continue-btn" href="${weeks[0] ? weeks[0].slug + '.html' : 'syllabus.html'}">Start with Week 1</a><a class="btn" href="syllabus.html">View the syllabus</a></div>
<div class="hero-stats"><div><b>${weeks.length}</b><span>weeks</span></div><div><b>${Math.round(totalWords / 1000)}k</b><span>words of notes</span></div><div><b>${totals.figures + totals.charts}</b><span>diagrams &amp; charts</span></div><div><b>${totals.demos}</b><span>interactive demos</span></div><div><b>${totals.exercises}</b><span>exercises with solutions</span></div></div>
</div>
<div class="hero-art" aria-hidden="true">${heroArt()}</div>
</section>

<section class="pillars">
<div class="pillar"><div class="pillar-i">∑</div><h3>Math you can follow</h3><p>Derivations are done step by step, with the intuition first and every symbol defined. Backpropagation, ELBOs, attention, policy gradients: nothing is left as an exercise to the reader.</p></div>
<div class="pillar"><div class="pillar-i">{ }</div><h3>Code that runs</h3><p>From-scratch NumPy to idiomatic PyTorch and Hugging Face. Every lab is small enough for a free Colab GPU and complete enough to learn from.</p></div>
<div class="pillar"><div class="pillar-i">◐</div><h3>Interactive by design</h3><p>Drag a learning rate, step through attention, push a GAN into mode collapse. Demos run entirely in your browser, offline and without accounts.</p></div>
<div class="pillar"><div class="pillar-i">↗</div><h3>From basics to the frontier</h3><p>Perceptrons to Transformers, LLMs, diffusion, graph networks, deep RL, scaling, safety and theory, in one coherent path.</p></div>
</section>

<section class="section"><div class="section-head"><h2>The learning path</h2><p>Four phases, each building on the last.</p></div><div class="phase-grid">${phaseCards}</div></section>

<section class="section"><div class="section-head"><h2>All weeks</h2><p>Jump to any chapter. Your progress is saved in this browser.</p></div><div class="week-grid">${cards}</div></section>

<section class="section split">
<div><h2>How to study with this course</h2>
<ol class="steps"><li><b>Read actively.</b> Try each derivation on paper before opening the collapsed solution.</li><li><b>Play with every demo.</b> Change one thing at a time and predict the outcome before you look.</li><li><b>Run the labs.</b> Copy the code into Colab or a local environment and break it on purpose.</li><li><b>Do the exercises.</b> They are graded from warm-up to research-flavoured; the hard ones are where learning happens.</li><li><b>Read the papers.</b> Each week ends with essential reading marked. Skim, then read closely.</li></ol></div>
<div class="note-card"><h3>Prerequisites</h3><p>Linear algebra, multivariate calculus, probability, Python/NumPy and an introductory machine learning course. Week 1 contains a compact math refresher.</p><h3>Setup</h3><p>Python 3.10+, PyTorch, and a free Google Colab account are enough for every lab.</p><a class="btn btn-small" href="about.html">Contribute or reuse this course</a></div>
</section>
</div>`;
  return shell({ nav, slug: 'index', title: `${SITE.name}: an open MS-level course`, desc: SITE.tagline, body, toc: [], hasChart: false, demos: [] });
}

function heroArt() {
  // Decorative network + loss-surface motif, drawn with theme variables.
  let nodes = '';
  const layers = [4, 6, 6, 3];
  const xs = [60, 190, 320, 450];
  const pos = layers.map((n, li) => Array.from({ length: n }, (_, i) => [xs[li], 210 + (i - (n - 1) / 2) * 46]));
  let lines = '';
  for (let l = 0; l < layers.length - 1; l++) for (const a of pos[l]) for (const b of pos[l + 1]) lines += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}"/>`;
  pos.forEach((layer, li) => layer.forEach((p, i) => (nodes += `<circle cx="${p[0]}" cy="${p[1]}" r="9" class="${(li + i) % 5 === 0 ? 'hn hot' : 'hn'}"/>`)));
  return `<svg viewBox="0 0 510 420" class="hero-svg" role="presentation"><g class="hl" stroke-width="1">${lines}</g>${nodes}<path class="hcurve" d="M20 372 C 90 372 110 330 170 330 S 260 392 330 366 S 430 300 490 312" fill="none" stroke-width="2.5" stroke-linecap="round"/><circle class="hdot" cx="330" cy="366" r="6"/></svg>`;
}

function notFound({ nav }) {
  const body = `<article class="article"><header class="article-head"><div class="eyebrow">404</div><h1>Page not found</h1><p class="lede">That page does not exist, or it has not been written yet.</p></header><p><a class="btn btn-primary" href="index.html">Back to the course</a></p></article>`;
  return shell({ nav, slug: '404', title: `Not found | ${SITE.name}`, desc: '', body, toc: [], hasChart: false, demos: [] });
}

module.exports = { week, doc, home, notFound, SITE };
