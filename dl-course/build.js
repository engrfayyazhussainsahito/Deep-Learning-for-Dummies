#!/usr/bin/env node
'use strict';
/*
 * Deep Learning Course site builder.
 *
 *   node build.js              full build into ./site
 *   node build.js --check=N    render only week N, print warnings + stats, write nothing
 *
 * Authoring conventions are documented in SPEC.md.
 */
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');
const MarkdownIt = require('markdown-it');
const anchor = require('markdown-it-anchor');
const katex = require('katex');
const hljs = require('highlight.js');

const ROOT = __dirname;
const SRC = path.join(ROOT, 'src');
const CONTENT = path.join(ROOT, 'content');
const FIGS = path.join(ROOT, 'figures');
const DEMOS = path.join(ROOT, 'demos');
const OUT = path.join(ROOT, 'site');

const PHASES = [
  { id: 1, name: 'Foundations', blurb: 'Math, neural networks, backpropagation, optimization and regularization.', weeks: [1, 2, 3, 4] },
  { id: 2, name: 'Core Architectures', blurb: 'CNNs, modern vision models, sequence models and representation learning.', weeks: [5, 6, 7, 8] },
  { id: 3, name: 'Attention & Pretrained Models', blurb: 'Transformers, large language models, vision-language models and generative models.', weeks: [9, 10, 11, 12] },
  { id: 4, name: 'Advanced Topics', blurb: 'Diffusion, graph networks, reinforcement learning, scale, trust and theory.', weeks: [13, 14, 15, 16, 17, 18] },
];

const KATEX_MACROS = {
  '\\R': '\\mathbb{R}',
  '\\N': '\\mathbb{N}',
  '\\E': '\\mathbb{E}',
  '\\P': '\\mathbb{P}',
  '\\argmax': '\\operatorname*{arg\\,max}',
  '\\argmin': '\\operatorname*{arg\\,min}',
  '\\norm': '\\left\\lVert #1 \\right\\rVert',
  '\\abs': '\\left\\lvert #1 \\right\\rvert',
  '\\softmax': '\\operatorname{softmax}',
  '\\relu': '\\operatorname{ReLU}',
  '\\KL': 'D_{\\mathrm{KL}}',
  '\\grad': '\\nabla',
  '\\diag': '\\operatorname{diag}',
  '\\tr': '\\operatorname{tr}',
  '\\sigmoid': '\\sigma',
  '\\Var': '\\operatorname{Var}',
  '\\Cov': '\\operatorname{Cov}',
  '\\ind': '\\mathbb{1}',
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const pad2 = (n) => String(n).padStart(2, '0');

function slugify(s) {
  return String(s)
    .toLowerCase()
    .replace(/<[^>]*>/g, '')
    .replace(/&[a-z]+;/g, '')
    .replace(/[^\p{L}\p{N}\s-]/gu, '')
    .trim()
    .replace(/\s+/g, '-');
}

function parseFront(src) {
  const m = src.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  const meta = {};
  if (m) {
    for (const line of m[1].split(/\r?\n/)) {
      const i = line.indexOf(':');
      if (i > 0) meta[line.slice(0, i).trim()] = line.slice(i + 1).trim().replace(/^["']|["']$/g, '');
    }
  }
  return { meta, body: m ? src.slice(m[0].length) : src };
}

function copyDir(src, dst, filter) {
  if (!fs.existsSync(src)) return;
  fs.mkdirSync(dst, { recursive: true });
  for (const e of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, e.name);
    const d = path.join(dst, e.name);
    if (e.isDirectory()) copyDir(s, d, filter);
    else if (!filter || filter(s)) fs.copyFileSync(s, d);
  }
}

// ---------------------------------------------------------------------------
// markdown engine
// ---------------------------------------------------------------------------
function makeMd() {
  const md = new MarkdownIt({ html: true, linkify: true, typographer: false });
  md.use(anchor, { slugify, tabIndex: false });
  // Support "## Heading text {#custom-id}": strip the marker and use it as the heading id.
  md.core.ruler.before('anchor', 'heading_custom_ids', (state) => {
    const toks = state.tokens;
    for (let i = 0; i < toks.length; i++) {
      if (toks[i].type !== 'heading_open') continue;
      const inl = toks[i + 1];
      if (!inl || inl.type !== 'inline') continue;
      const m = inl.content.match(/\s*\{#([A-Za-z0-9_-]+)\}\s*$/);
      if (!m) continue;
      toks[i].attrSet('id', m[1]);
      inl.content = inl.content.slice(0, m.index);
      const kids = inl.children || [];
      for (let k = kids.length - 1; k >= 0; k--) {
        if (kids[k].type === 'text' && /\{#[A-Za-z0-9_-]+\}\s*$/.test(kids[k].content)) {
          kids[k].content = kids[k].content.replace(/\s*\{#[A-Za-z0-9_-]+\}\s*$/, '');
          break;
        }
      }
    }
  });

  md.renderer.rules.fence = (tokens, idx, options, env) => {
    const t = tokens[idx];
    const info = (t.info || '').trim();
    const colon = info.indexOf(':');
    const lang = (colon >= 0 ? info.slice(0, colon) : info).trim().toLowerCase();
    const title = colon >= 0 ? info.slice(colon + 1).trim() : '';
    const code = t.content.replace(/\n$/, '');

    if (lang === 'chart') {
      let cfg;
      try {
        cfg = JSON.parse(code);
      } catch (e) {
        env.warn(`invalid chart JSON: ${e.message}`);
        return `<div class="callout callout-warning"><div class="callout-title">Chart error</div><p>${esc(e.message)}</p></div>`;
      }
      env.stats.charts++;
      const caption = cfg.caption || '';
      const height = Math.min(Math.max(parseInt(cfg.height || 340, 10) || 340, 180), 700);
      delete cfg.caption;
      delete cfg.height;
      if (!cfg.type || !cfg.data) env.warn('chart needs "type" and "data"');
      env.figN++;
      const label = `Figure ${env.figPrefix}${env.figN}`;
      return (
        `<figure class="chart-fig"><div class="chart-wrap" style="height:${height}px">` +
        `<canvas role="img" aria-label="${esc(caption || 'chart')}" data-chart="${esc(JSON.stringify(cfg))}"></canvas></div>` +
        `<figcaption><strong>${label}.</strong> ${env.inline(caption)}</figcaption></figure>`
      );
    }

    if (lang === 'python' || lang === 'py') env.pyBlocks.push({ code, title });
    env.stats.code++;
    let html;
    if (lang && hljs.getLanguage(lang)) html = hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
    else html = esc(code);
    const shownLang = lang === 'py' ? 'python' : lang || 'text';
    return (
      `<div class="codeblock" data-lang="${esc(shownLang)}"><div class="codeblock-head">` +
      `<span class="codeblock-lang">${esc(shownLang)}</span>` +
      (title ? `<span class="codeblock-title">${esc(title)}</span>` : '') +
      `<button class="copy-btn" type="button" aria-label="Copy code">Copy</button></div>` +
      `<pre><code class="hljs language-${esc(shownLang)}">${html}</code></pre></div>\n`
    );
  };

  md.renderer.rules.table_open = () => '<div class="table-wrap"><table>\n';
  md.renderer.rules.table_close = () => '</table></div>\n';

  const defaultLink = md.renderer.rules.link_open || ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options));
  md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
    const href = tokens[idx].attrGet('href') || '';
    if (/^https?:\/\//.test(href)) {
      tokens[idx].attrSet('target', '_blank');
      tokens[idx].attrSet('rel', 'noopener');
    }
    return defaultLink(tokens, idx, options, env, self);
  };
  return md;
}

const MD = makeMd();

const CALLOUT_LABELS = {
  note: 'Note',
  tip: 'Tip',
  warning: 'Watch out',
  intuition: 'Intuition',
  math: 'Derivation',
  key: 'Key takeaways',
  exercise: 'Exercise',
  definition: 'Definition',
  example: 'Worked example',
  history: 'History',
  details: 'Details',
};

function renderMarkdown(body, ctx) {
  const warn = (m) => ctx.warnings.push(m);
  const stats = ctx.stats;
  const mathStore = [];
  const codeStore = [];
  const figStore = [];
  const demoStore = [];
  let text = body.replace(/\r\n/g, '\n');

  // 1. protect code
  text = text.replace(/^[ \t]*(```|~~~)[^\n]*\n[\s\S]*?^[ \t]*\1[ \t]*$/gm, (m) => {
    codeStore.push(m);
    return `CODEPH${codeStore.length - 1}END`;
  });
  text = text.replace(/`[^`\n]+`/g, (m) => {
    codeStore.push(m);
    return `CODEPH${codeStore.length - 1}END`;
  });

  // 2. math
  const renderTex = (tex, display) => {
    try {
      return katex.renderToString(tex.trim(), { displayMode: display, throwOnError: true, macros: { ...KATEX_MACROS }, strict: 'ignore' });
    } catch (e) {
      warn(`KaTeX error in "${tex.trim().slice(0, 60)}": ${e.message.slice(0, 120)}`);
      return `<span class="math-error">${esc(tex)}</span>`;
    }
  };
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (m, tex) => {
    stats.math++;
    mathStore.push(renderTex(tex, true));
    return `MATHPH${mathStore.length - 1}END`;
  });
  text = text.replace(/(?<![\\$])\$(?![\s$])((?:[^$\n\\]|\\.)+?)(?<![\s\\])\$(?!\d)/g, (m, tex) => {
    stats.math++;
    mathStore.push(renderTex(tex, false));
    return `MATHPH${mathStore.length - 1}END`;
  });
  text = text.replace(/\\\$/g, '$');

  // headings must not contain math (slugs would break)
  for (const line of text.split('\n')) {
    if (/^#{1,6}\s/.test(line) && /MATHPH\d+END/.test(line)) warn(`math in heading (avoid): ${line.slice(0, 70)}`);
  }

  // 3. directives
  text = text.replace(/^\{\{figure:\s*([^|}\n]+?)\s*(?:\|\s*([^\n]*?))?\s*\}\}\s*$/gm, (m, file, caption) => {
    figStore.push({ file: file.trim(), caption: caption || '' });
    return `\n\nFIGPH${figStore.length - 1}END\n\n`;
  });
  text = text.replace(/^\{\{demo:\s*([\w-]+)\s*(?:\|\s*([^|}\n]*?))?\s*(?:\|\s*(\d+))?\s*\}\}\s*$/gm, (m, name, title, h) => {
    demoStore.push({ name, title: title || '', height: h ? parseInt(h, 10) : 0 });
    return `\n\nDEMOPH${demoStore.length - 1}END\n\n`;
  });

  // 4. callouts (stack based, supports nesting)
  const out = [];
  const stack = [];
  for (const line of text.split('\n')) {
    const open = line.match(/^:::\s*(\w+)(?:\s+(.*))?$/);
    if (open) {
      const type = open[1].toLowerCase();
      const title = (open[2] || '').trim();
      if (!CALLOUT_LABELS[type]) warn(`unknown callout type ":::${type}"`);
      stats.callouts++;
      if (type === 'exercise') stats.exercises++;
      const label = title || CALLOUT_LABELS[type] || type;
      if (type === 'details') {
        stack.push('details');
        out.push('', `<details class="callout callout-details"><summary>${label}</summary>`, '');
      } else {
        stack.push('div');
        out.push('', `<div class="callout callout-${type}"><div class="callout-title">${label}</div>`, '');
      }
      continue;
    }
    if (/^:::\s*$/.test(line)) {
      const k = stack.pop();
      if (!k) warn('stray ::: closer');
      else out.push('', `</${k}>`, '');
      continue;
    }
    out.push(line);
  }
  if (stack.length) warn(`${stack.length} unclosed ::: callout(s)`);
  text = out.join('\n');

  // 5. restore code
  text = text.replace(/CODEPH(\d+)END/g, (m, i) => codeStore[+i]);

  // 6. render
  const env = {
    warn,
    stats,
    pyBlocks: ctx.pyBlocks,
    figN: 0,
    figPrefix: ctx.figPrefix || '',
    inline: (s) => MD.renderInline(s || '', {}),
  };
  let html = MD.render(text, env);

  // 7. figures / demos
  html = html.replace(/<p>FIGPH(\d+)END<\/p>/g, (m, i) => {
    const f = figStore[+i];
    const file = path.join(FIGS, f.file.endsWith('.svg') ? f.file : f.file + '.svg');
    if (!fs.existsSync(file)) {
      warn(`missing figure: ${f.file}`);
      return `<div class="callout callout-warning"><div class="callout-title">Missing figure</div><p>${esc(f.file)}</p></div>`;
    }
    let svg = fs.readFileSync(file, 'utf8').replace(/<\?xml[\s\S]*?\?>/g, '').replace(/<!DOCTYPE[\s\S]*?>/gi, '').trim();
    if (/<script/i.test(svg)) {
      warn(`figure ${f.file} contains <script> (not allowed)`);
      svg = svg.replace(/<script[\s\S]*?<\/script>/gi, '');
    }
    if (!/viewBox=/.test(svg)) warn(`figure ${f.file} has no viewBox`);
    const alt = f.caption.replace(/MATHPH\d+END/g, '').replace(/[*_`]/g, '') || f.file;
    svg = svg.replace(/<svg\b([^>]*)>/, (mm, attrs) => {
      let a = attrs.replace(/\s(width|height)="[^"]*"/g, '');
      if (!/role=/.test(a)) a += ' role="img"';
      if (!/aria-label=/.test(a)) a += ` aria-label="${esc(alt)}"`;
      if (!/class=/.test(a)) a += ' class="fig-svg"';
      return `<svg${a}>`;
    });
    env.figN++;
    stats.figures++;
    ctx.figs.push(f.file);
    const label = `Figure ${env.figPrefix}${env.figN}`;
    return (
      `<figure class="figure"><div class="figure-art">${svg}</div>` +
      `<figcaption><strong>${label}.</strong> ${f.caption ? env.inline(f.caption) : ''}</figcaption></figure>`
    );
  });
  html = html.replace(/<p>DEMOPH(\d+)END<\/p>/g, (m, i) => {
    const d = demoStore[+i];
    if (!fs.existsSync(path.join(DEMOS, d.name + '.js'))) warn(`missing demo script: demos/${d.name}.js`);
    stats.demos++;
    ctx.demos.add(d.name);
    const style = d.height ? ` style="min-height:${d.height}px"` : '';
    return (
      `<div class="demo" data-demo="${esc(d.name)}"><div class="demo-head"><span class="demo-badge">Interactive</span>` +
      `<span class="demo-title">${env.inline(d.title || d.name)}</span></div><div class="demo-body"${style}><noscript>Enable JavaScript to use this demo.</noscript></div></div>`
    );
  });

  // 8. math
  html = html.replace(/MATHPH(\d+)END/g, (m, i) => mathStore[+i]);
  if (/(MATHPH|CODEPH|FIGPH|DEMOPH)\d+END/.test(html)) warn('unresolved placeholder in output (check directive syntax)');

  // toc
  const toc = [];
  const re = /<h([23]) id="([^"]+)"[^>]*>([\s\S]*?)<\/h\1>/g;
  let m;
  while ((m = re.exec(html))) toc.push({ level: +m[1], id: m[2], text: m[3].replace(/<[^>]+>/g, '') });

  return html;
}

function plainText(body) {
  return body
    .replace(/^[ \t]*(```|~~~)[^\n]*\n[\s\S]*?^[ \t]*\1[ \t]*$/gm, ' ')
    .replace(/\$\$[\s\S]+?\$\$/g, ' ')
    .replace(/(?<![\\$])\$(?![\s$])(?:[^$\n\\]|\\.)+?(?<![\s\\])\$(?!\d)/g, ' ')
    .replace(/^\{\{(figure|demo):[^|}\n]*\|?\s*([^}\n]*)\}\}\s*$/gm, '$2')
    .replace(/^:::\s*\w*\s*/gm, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/[*_`>#|]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function countWords(body) {
  const t = plainText(body);
  return t ? t.split(/\s+/).length : 0;
}

function extractToc(html) {
  const toc = [];
  const re = /<h([23]) id="([^"]+)"[^>]*>([\s\S]*?)<\/h\1>/g;
  let m;
  while ((m = re.exec(html))) toc.push({ level: +m[1], id: m[2], text: m[3].replace(/<[^>]+>/g, '').replace(/&amp;/g, '&') });
  return toc;
}

function checkPython(blocks, warn) {
  let bad = 0;
  for (const b of blocks) {
    const r = spawnSync('python3', ['-c', 'import ast,sys; ast.parse(sys.stdin.read())'], { input: b.code, encoding: 'utf8' });
    if (r.status !== 0) {
      bad++;
      const lines = (r.stderr || '').trim().split('\n');
      warn(`python syntax error in block starting "${b.code.split('\n')[0].slice(0, 50)}": ${lines[lines.length - 1]}`);
    }
  }
  return bad;
}

function loadPage(file) {
  const raw = fs.readFileSync(file, 'utf8');
  const { meta, body } = parseFront(raw);
  return { file, meta, body };
}

function renderPage(page, kind) {
  const ctx = {
    warnings: [],
    stats: { math: 0, code: 0, figures: 0, charts: 0, demos: 0, callouts: 0, exercises: 0 },
    pyBlocks: [],
    figs: [],
    demos: new Set(),
    figPrefix: kind === 'week' ? `${page.meta.week}.` : '',
  };
  const html = renderMarkdown(page.body, ctx);
  checkPython(ctx.pyBlocks, (m) => ctx.warnings.push(m));
  const words = countWords(page.body);
  return { ctx, html, words, toc: extractToc(html) };
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
function listWeeks() {
  if (!fs.existsSync(CONTENT)) return [];
  return fs
    .readdirSync(CONTENT)
    .filter((f) => /^week-\d\d\.md$/.test(f))
    .sort()
    .map((f) => loadPage(path.join(CONTENT, f)))
    .map((p) => {
      p.meta.week = String(parseInt(path.basename(p.file).slice(5, 7), 10));
      return p;
    });
}

function checkMode(n) {
  const file = path.join(CONTENT, `week-${pad2(n)}.md`);
  if (!fs.existsSync(file)) {
    console.error(`content/week-${pad2(n)}.md not found`);
    process.exit(1);
  }
  const page = loadPage(file);
  page.meta.week = String(n);
  const r = renderPage(page, 'week');
  const s = r.ctx.stats;
  console.log(`Week ${n}: "${page.meta.title || '(no title)'}"`);
  console.log(`  words: ${r.words}   math: ${s.math}   code blocks: ${s.code}   figures: ${s.figures}   charts: ${s.charts}   demos: ${s.demos}   callouts: ${s.callouts}   exercises: ${s.exercises}`);
  console.log(`  sections (h2): ${r.toc.filter((t) => t.level === 2).length}   subsections (h3): ${r.toc.filter((t) => t.level === 3).length}`);
  for (const k of ['title', 'summary', 'phase']) if (!page.meta[k]) r.ctx.warnings.push(`front matter missing "${k}"`);
  if (r.ctx.warnings.length) {
    console.log(`  WARNINGS (${r.ctx.warnings.length}):`);
    for (const w of r.ctx.warnings) console.log('   - ' + w);
    process.exitCode = 2;
  } else console.log('  no warnings');
}

// ---------------------------------------------------------------------------
// internal link resolver: repairs anchors that do not exist by fuzzy-matching
// them against the real heading ids of the target page (or drops the anchor).
// ---------------------------------------------------------------------------
function resolveLinks(pages) {
  const headingIds = {};
  for (const [name, html] of Object.entries(pages)) {
    headingIds[name] = [...html.matchAll(/<h[1-6][^>]*\sid="([^"]+)"/g)].map((m) => m[1]);
  }
  const toks = (s) => s.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  const stem = (t) => t.replace(/(ing|ed|es|s)$/, '').slice(0, 6);
  const score = (a, b) => {
    const A = toks(a).map(stem), B = toks(b).map(stem);
    if (!A.length || !B.length) return 0;
    let hit = 0;
    for (const x of A) if (B.some((y) => y === x || (x.length > 3 && y.length > 3 && (y.startsWith(x) || x.startsWith(y))))) hit++;
    return hit / Math.max(A.length, B.length) * 0.6 + hit / A.length * 0.4;
  };
  const best = (anchor, ids) => {
    let top = null, ts = 0;
    for (const id of ids) {
      const sc = score(anchor, id);
      if (sc > ts) { ts = sc; top = id; }
    }
    return ts >= 0.5 ? top : null;
  };
  const log = { fixed: 0, dropped: 0, details: [] };
  for (const [name, html] of Object.entries(pages)) {
    pages[name] = html.replace(/href="([^"#]*)#([^"]+)"/g, (m, file, anchor) => {
      if (/^https?:/.test(file)) return m;
      const target = file || name;
      if (!pages[target]) return m;
      const ids = headingIds[target];
      if (ids.includes(anchor) || new RegExp(`\\sid="${anchor.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"`).test(pages[target])) return m;
      const fix = best(anchor, ids);
      if (fix) {
        log.fixed++;
        log.details.push(`${name}: ${file}#${anchor} -> #${fix}`);
        return `href="${file}#${fix}"`;
      }
      log.dropped++;
      log.details.push(`${name}: ${file}#${anchor} -> (anchor dropped)`);
      return `href="${file || name}"`;
    });
  }
  return log;
}

function build() {
  const template = require('./src/template.js');
  fs.mkdirSync(OUT, { recursive: true });

  // assets
  copyDir(path.join(SRC, 'assets'), path.join(OUT, 'assets'));
  const vendor = path.join(OUT, 'assets', 'vendor');
  fs.mkdirSync(vendor, { recursive: true });
  const nm = path.join(ROOT, 'node_modules');
  fs.mkdirSync(path.join(vendor, 'katex'), { recursive: true });
  fs.copyFileSync(path.join(nm, 'katex/dist/katex.min.css'), path.join(vendor, 'katex/katex.min.css'));
  copyDir(path.join(nm, 'katex/dist/fonts'), path.join(vendor, 'katex/fonts'), (f) => f.endsWith('.woff2'));
  fs.copyFileSync(path.join(nm, 'chart.js/dist/chart.umd.js'), path.join(vendor, 'chart.umd.js'));
  const fontDir = path.join(OUT, 'assets', 'fonts');
  fs.mkdirSync(fontDir, { recursive: true });
  const fonts = [
    ['@fontsource-variable/inter/files/inter-latin-wght-normal.woff2', 'inter-var.woff2'],
    ['@fontsource-variable/source-serif-4/files/source-serif-4-latin-wght-normal.woff2', 'source-serif-var.woff2'],
    ['@fontsource-variable/source-serif-4/files/source-serif-4-latin-wght-italic.woff2', 'source-serif-var-italic.woff2'],
    ['@fontsource/jetbrains-mono/files/jetbrains-mono-latin-400-normal.woff2', 'jetbrains-400.woff2'],
    ['@fontsource/jetbrains-mono/files/jetbrains-mono-latin-500-normal.woff2', 'jetbrains-500.woff2'],
    ['@fontsource/jetbrains-mono/files/jetbrains-mono-latin-700-normal.woff2', 'jetbrains-700.woff2'],
  ];
  for (const [from, to] of fonts) fs.copyFileSync(path.join(nm, from), path.join(fontDir, to));
  copyDir(DEMOS, path.join(OUT, 'assets', 'demos'), (f) => f.endsWith('.js'));

  // pages
  const weeks = listWeeks();
  const weekInfo = weeks.map((p) => {
    const n = parseInt(p.meta.week, 10);
    return { n, slug: `week-${pad2(n)}`, title: p.meta.title || `Week ${n}`, summary: p.meta.summary || '', phase: p.meta.phase || '', page: p };
  });
  const navData = { phases: PHASES, weeks: weekInfo.map(({ n, slug, title, summary }) => ({ n, slug, title, summary })) };

  const searchIndex = [];
  const report = [];
  let totalWords = 0;
  const totals = { figures: 0, charts: 0, demos: 0, exercises: 0, code: 0, math: 0 };
  let totalWarn = 0;

  const outPages = {};
  const emit = (name, html) => { outPages[name] = html; };

  weekInfo.forEach((w, i) => {
    const r = renderPage(w.page, 'week');
    const minutes = Math.max(5, Math.round(r.words / 210 + r.ctx.stats.demos * 3 + r.ctx.stats.exercises * 2));
    totalWords += r.words;
    for (const k of Object.keys(totals)) totals[k] += r.ctx.stats[k] || 0;
    totalWarn += r.ctx.warnings.length;
    const demos = [...r.ctx.demos];
    const hasChart = r.ctx.stats.charts > 0;
    emit(
      `${w.slug}.html`,
      template.week({
        nav: navData,
        week: w,
        prev: weekInfo[i - 1] || null,
        next: weekInfo[i + 1] || null,
        html: r.html,
        toc: r.toc,
        minutes,
        stats: r.ctx.stats,
        words: r.words,
        demos,
        hasChart,
        phase: PHASES.find((p) => p.weeks.includes(w.n)),
      })
    );
    // search sections
    const parts = w.page.body.split(/^## +/m);
    parts.forEach((part, idx) => {
      const rawHead = idx === 0 ? w.title : part.split('\n')[0].trim();
      const idm = rawHead.match(/\{#([A-Za-z0-9_-]+)\}\s*$/);
      const heading = rawHead.replace(/\s*\{#[A-Za-z0-9_-]+\}\s*$/, '').replace(/[*_`$]/g, '');
      const t = plainText(idx === 0 ? part : part.split('\n').slice(1).join('\n'));
      if (t.length < 40) return;
      searchIndex.push({ p: `${w.slug}.html`, w: `Week ${w.n}`, t: w.title, h: heading, a: idx === 0 ? '' : (idm ? idm[1] : slugify(heading)), x: t.slice(0, 1800) });
    });
    report.push(`week ${pad2(w.n)}: ${r.words} words, ${r.ctx.stats.figures} figs, ${r.ctx.stats.charts} charts, ${r.ctx.stats.demos} demos, ${r.ctx.stats.code} code, ${r.ctx.stats.exercises} ex, ${r.ctx.warnings.length} warnings`);
    for (const wn of r.ctx.warnings) report.push(`   ! ${wn}`);
  });

  // doc pages
  const docs = fs.existsSync(CONTENT) ? fs.readdirSync(CONTENT).filter((f) => f.endsWith('.md') && !/^week-/.test(f)) : [];
  for (const f of docs) {
    const p = loadPage(path.join(CONTENT, f));
    const slug = f.replace(/\.md$/, '');
    const r = renderPage(p, 'doc');
    totalWarn += r.ctx.warnings.length;
    for (const wn of r.ctx.warnings) report.push(`${slug}: ! ${wn}`);
    emit(`${slug}.html`, template.doc({ nav: navData, slug, meta: p.meta, html: r.html, toc: r.toc, hasChart: r.ctx.stats.charts > 0, demos: [...r.ctx.demos] }));
    const t = plainText(p.body);
    searchIndex.push({ p: `${slug}.html`, w: 'Page', t: p.meta.title || slug, h: p.meta.title || slug, a: '', x: t.slice(0, 1800) });
  }

  // home
  emit('index.html', template.home({ nav: navData, weeks: weekInfo, totalWords, totals }));
  emit('404.html', template.notFound({ nav: navData }));

  const linkLog = resolveLinks(outPages);
  for (const [name, html] of Object.entries(outPages)) fs.writeFileSync(path.join(OUT, name), html);
  fs.writeFileSync(path.join(ROOT, 'link-report.txt'), linkLog.details.join('\n') + '\n');
  console.log(`links: ${linkLog.fixed} anchors repaired by fuzzy match, ${linkLog.dropped} dropped to page level (see link-report.txt)`);
  fs.writeFileSync(path.join(OUT, 'assets', 'search-index.js'), 'window.DL_SEARCH=' + JSON.stringify(searchIndex) + ';');
  fs.writeFileSync(path.join(OUT, '.nojekyll'), '');

  console.log(report.join('\n'));
  console.log(`\nBuilt ${weekInfo.length} weeks + ${docs.length} pages, ${totalWords} words total, ${totalWarn} warnings -> ${path.relative(process.cwd(), OUT) || OUT}`);
  if (totalWarn) process.exitCode = 2;
}

const checkArg = process.argv.find((a) => a.startsWith('--check='));
if (checkArg) checkMode(parseInt(checkArg.split('=')[1], 10));
else build();
