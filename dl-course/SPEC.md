# Authoring Spec: Deep Learning (MS level) open course website

You are writing study material for a **free, open-source, world-class** online Deep Learning course
(MS level, 18 weeks). Think: a textbook chapter crossed with Distill.pub, Stanford CS231n/CS224n notes,
the d2l.ai book, and Karpathy's lectures. Students will study from these pages alone, so every topic in
the week's syllabus scope (see `content/syllabus.md`, section "Weekly Schedule") must be covered **in depth**,
from intuition to math to code to pitfalls.

Project root: the repository root. When working in parallel, only touch YOUR files (named with your week prefixes). Never edit
`build.js`, `src/`, `SPEC.md`, or another author's files. If you find a bug in the framework, mention it in your
final message and work around it.

## Files you create (for week NN, two-digit)

| What | Where |
| --- | --- |
| Page content (Markdown) | `content/week-NN.md` |
| Diagrams (SVG) | `figures/weekNN-<name>.svg` |
| Interactive demos (JS) | `demos/weekNN-<name>.js` |

## Verify as you go

```
node build.js --check=N      # renders only week N; prints stats + warnings, writes nothing
```

Fix every warning (KaTeX errors, missing figure/demo, bad chart JSON, python syntax errors, unclosed callouts).
Write long files in several steps (Write the first part, then append with `cat >> file <<'EOF'` using a quoted
heredoc). Python 3 is available. `pip install numpy matplotlib` works; PyTorch is NOT installed (don't try to
install it). You can run NumPy-only code to verify it and to **generate real data for charts** (e.g. loss
curves from a real toy training run). Use WebSearch to verify paper titles, authors, years and facts you are not
100% sure about. Never invent citations, numbers or quotes. Do not copy text from books/papers/websites: write
everything in your own words (short quotes under 15 words only if essential).

## Quality bar per week (minimums, exceed them where the topic deserves it)

- **Length:** 7,000+ words of explanatory prose (code and math excluded), 12+ H2/H3 sections. Depth over padding.
- **Coverage:** every bullet of the week's syllabus scope gets its own H2 or H3 with a full treatment.
- **Figures:** 6+ original SVG diagrams (architectures, computation graphs, geometry, data flow, timelines).
- **Charts:** 3+ data charts (`chart` blocks) using real numbers (computed by you, or well-known published
  values with the source named in the caption; label anything illustrative as "illustrative").
- **Interactive demos:** 3+ distinct, polished, working demos (see below). Each must teach something that a
  static page cannot (sliders that change a curve, step-through animations, playgrounds).
- **Code:** 8+ complete code blocks. PyTorch is the main framework (`import torch`), plus from-scratch NumPy where the
  point is to understand the mechanism. Code must be correct, idiomatic, commented, and runnable top-to-bottom in
  Colab (use small datasets: MNIST, CIFAR-10, toy data, tiny corpora). Include at least one end-to-end
  "Lab" section that walks through the week's lab with full code. Use realistic shapes and print/inspect tensors.
- **Math:** derive the important results step by step (use `:::math` derivation callouts, collapsed content is fine
  with `:::details` inside). Define every symbol. Give intuition BEFORE the formula.
- **Exercises:** 8+ exercises (mix of derivation, coding, conceptual, and "what would happen if..."), each with a
  collapsible solution (`:::details Solution`). At least two should be challenging.
- **Pitfalls:** a "Common mistakes and debugging tips" section with concrete failure symptoms and fixes.
- **History/context:** brief, accurate notes on where ideas came from (who, when, why it mattered).
- **Readings:** "Further reading" list of real, verifiable papers/books/blog posts: authors, title, year, venue or
  arXiv id. Mark the 3 or so essential ones.
- **Voice:** clear, precise, friendly, rigorous. Address the student as "you". Prefer concrete examples and
  numbers over vague claims. Explain *why* and *when it fails*, not just *what*.

## Page structure (suggested)

```markdown
---
title: Optimization for Deep Learning
phase: Foundations
summary: One or two sentences (shown on cards and in search).
---

:::key Learning objectives
- ...
:::

Short motivating intro (no H1: the page template renders the title).

## Section
...
## Lab: ...
## Exercises
## Key takeaways
:::key Key takeaways
...
:::
## Further reading
```

Front matter keys: `title`, `phase` (one of `Foundations`, `Core Architectures`, `Attention & Pretrained Models`,
`Advanced Topics`), `summary`. The week number comes from the file name. Do NOT put `$...$` math inside headings.

## Markdown features

GitHub-flavoured Markdown (tables, lists, bold, links) plus:

### Math (KaTeX)
Inline `$x^2$`, display `$$ ... $$` (own lines). Escape a literal dollar as `\$`. Available macros:
`\R \N \E \P \argmax \argmin \norm{x} \abs{x} \softmax \relu \KL \grad \diag \tr \Var \Cov \ind`.
Use `\boldsymbol{w}` for vectors, `\mathbf{W}` for matrices, `\mathcal{L}` for loss. Align multi-line derivations
with `\begin{aligned} ... \end{aligned}`.

### Callouts
```
:::type Optional custom title
Markdown content (any blocks, including code and math)
:::
```
Types: `note`, `tip`, `warning`, `intuition`, `math` (derivations), `key`, `exercise`, `definition`, `example`,
`history`, `details` (collapsible; use for solutions and long derivations). Callouts can nest
(`:::details Solution` inside `:::exercise`). Each `:::` closer must be on its own line.

### Code
Fenced blocks with a language: `python`, `bash`, `javascript`, `json`, `text`. Add a title with a colon:
```python:train.py
```
All `python` blocks are syntax-checked with `ast.parse`, so use `text` for pseudo-code or output. Show expected
output in a separate `text` block when useful.

### Figures (SVG diagrams)
On its own line: `{{figure: weekNN-name | Caption text (inline markdown + math allowed)}}`
This inlines `figures/weekNN-name.svg`. Rules for SVG files:

- Must have `viewBox` (e.g. `viewBox="0 0 720 360"`); NO width/height attributes, NO `<script>`, no external refs.
- Must work in light and dark themes: **never hard-code colors**. Use these CSS classes / variables:
  - Text: `t` (normal 13px), `t-sm`, `t-lg`, `t-bold`, `t-muted`, `t-mono`, `t-accent`; set alignment with
    `text-anchor="middle"` and `dominant-baseline="middle"`.
  - Shapes (rect/circle/ellipse/path): `box` (neutral), `box-accent`, `box-blue`, `box-green`, `box-purple`, `box-amber`,
    `box-red`, `box-teal` (tinted fill + colored stroke), `fill-blue` etc. (solid fill), `fill-soft`, `fill-ink`.
  - Lines/paths: `ln` (muted 1.5px), `ln-thin`, `ln-bold`, `ln-accent`, `ln-blue`, `ln-green`, `ln-purple`, `ln-amber`,
    `ln-red`, `ln-dash` (dashed). Add `fill="none"` on paths.
  - Variables available in inline `style`: `--c-ink --c-muted --c-line --c-bg --c-soft --c-accent --c-blue --c-green
    --c-purple --c-amber --c-red --c-teal` (e.g. `style="fill:var(--c-blue)"`).
  - Arrowheads: define a marker with a **file-unique id**, e.g.
    `<defs><marker id="w05a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" style="fill:var(--c-muted)"/></marker></defs>`
    then `<line class="ln" marker-end="url(#w05a)" .../>`.
- Keep text legible: nothing below 11px at the viewBox scale; leave margins; avoid overlap. Prefer 640 to 800 wide.
- Draw real, accurate diagrams (correct tensor shapes, correct arrows). Quality over quantity.

### Charts (Chart.js 4, JSON only, no functions)
````
```chart
{
  "type": "line",
  "caption": "Training loss for SGD vs Adam on a toy problem (computed with the NumPy code above).",
  "height": 340,
  "data": { "labels": [0,1,2,3], "datasets": [ { "label": "SGD", "data": [1.0, 0.6, 0.4, 0.3], "borderColor": "var(--c-blue)" } ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "step"}}, "y": {"type": "logarithmic"} } }
}
```
````
Types: `line`, `bar`, `scatter`, `radar`, `doughnut`, `bubble`. `caption` and `height` (px, 180 to 700) are extras that the
site strips. Colors may be `var(--c-blue)` etc. (resolved at runtime); if omitted, a palette is applied
automatically. Set `"pointRadius": 0` for dense lines, `"tension"` for smoothing, axis titles always.
For scatter use `"data": {"datasets":[{"label":"..","data":[{"x":1,"y":2}, ...]}]}`. Keep data under ~600 points per series.

### Interactive demos
On its own line: `{{demo: weekNN-name | Title shown above the demo | 420}}` (the optional last number is the minimum height in px).
Create `demos/weekNN-name.js`:

```javascript
DL.register('weekNN-name', function (root) {
  var ctrl = DL.controls(root);
  var lr = DL.slider(ctrl, { label: 'Learning rate', min: 0.001, max: 1, step: 0.001, value: 0.1 }, draw);
  var c = DL.canvas(root, { aspect: 0.5 });          // c.ctx, c.w, c.h, c.draw
  var P = DL.plot(c, { xmin: -3, xmax: 3, ymin: -1, ymax: 5, xlabel: 'x', ylabel: 'f(x)' });
  var say = DL.readout(root);
  function draw() {
    var col = DL.colors();                           // theme-aware colors, read at draw time
    P.clear(); P.grid(); P.axes();
    P.fn(function (x) { return x * x; }, col.accent, 2.5);
    say('lr = <b>' + DL.fmt(lr.get(), 3) + '</b>');
  }
  c.draw = draw;
});
```
Available helpers (see `src/assets/dl-lib.js` for exact signatures): `DL.el`, `DL.controls`, `DL.slider`, `DL.select`,
`DL.button`, `DL.toggle`, `DL.row`, `DL.readout`, `DL.canvas`, `DL.plot` (axes, grid, line, fn, points, text, hline,
vline, contour, heat), `DL.heatmap`, `DL.cmap.*`, `DL.colors`, `DL.onTheme`, `DL.rng`, `DL.randn`, `DL.softmax`,
`DL.matmul`, `DL.linspace`, `DL.drag`, etc. Rules: plain JavaScript (ES2015 OK), self-contained, no network, no
globals except `DL`; draw with colors from `DL.colors()` at draw time (so the dark theme works); handle resize via
`DL.canvas` (set `c.draw`); never throw on extreme slider values; animation loops (requestAnimationFrame) must pause
when the demo scrolls out of view or a Pause button is pressed, and must stop cleanly. Provide labelled controls, a
readout of the key numbers, and a short text hint of what to try. Test the numeric logic with `node` where possible
(e.g. stub the DOM) or by careful reasoning; correctness matters more than flashiness.
Math notation inside canvases/readouts uses Unicode (θ, ∇, σ, ∑, ², ₜ), since KaTeX is not available there.

## Accuracy and style rules

- Be technically exact: shapes, indices, signs, normalizations (e.g. 1/sqrt(d_k), pre- vs post-norm, biased vs
  unbiased estimators). If a claim is a rule of thumb, say so. Distinguish what is proven from what is empirical.
- Name real systems/papers correctly, with years. When you cite benchmark numbers, they must be correct and sourced.
- Avoid filler, avoid marketing language, avoid repeating the syllabus text. Teach.
- Cross-reference other weeks by plain links like `[Week 3](week-03.html)` or `[Optimization](week-03.html#adaptive-methods)`.
- Heading ids are lowercase-hyphenated slugs of the heading text.
- ASCII only is not required, but avoid fancy typography that breaks: use normal quotes and the `-` or `--` instead of em dashes.

## Final message to the coordinator

When done, run `node build.js --check=N` for each of your weeks and reply with: the files you created, the stats line
for each week, any remaining warnings, and any framework issues. Keep the message short.
