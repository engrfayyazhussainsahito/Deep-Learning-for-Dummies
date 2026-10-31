# Deep Learning: an open MS-level course (18 weeks)

A free, open-source, graduate-level course on deep learning, built as a static website you can
read online, run offline from a folder, or host anywhere. It goes from perceptrons and
backpropagation to Transformers, large language models, diffusion models, graph networks,
deep reinforcement learning, efficient training at scale, and trustworthy deep learning.

| | |
| --- | --- |
| Weeks | 18 (four phases: Foundations, Core Architectures, Attention and Pretrained Models, Advanced Topics) |
| Notes | about 208,000 words with step-by-step derivations |
| Visuals | 237 original diagrams and data charts |
| Interactive | 64 in-browser demos (no server, no accounts) |
| Exercises | 245 with collapsible worked solutions, plus practice midterm and final |
| Code | PyTorch, NumPy and Hugging Face examples and labs, sized for a free Colab GPU |
| Extras | Searchable text, glossary (190+ terms), dark mode, progress tracking, mobile friendly |

## Use it

**Read it right now (no install):** open `site/index.html` in a browser. Everything is prebuilt and works offline.

**Or serve it:**

```bash
cd site && python3 -m http.server 8000     # http://localhost:8000
```

## Rebuild from source

Requires Node.js 18+.

```bash
npm install
node build.js        # writes the website to site/
```

`node build.js --check=N` validates and renders a single week (math, figures, charts, demos,
Python syntax) without writing files.

## Deploy

`site/` is plain HTML/CSS/JS with relative paths, so any static host works (GitHub Pages,
Netlify, Cloudflare Pages, an S3 bucket, a university web server). A GitHub Actions workflow
for Pages is included in `.github/workflows/pages.yml`.

## Repository layout

```
content/    Markdown source: week-01.md ... week-18.md, syllabus.md, glossary.md, about.md
figures/    theme-aware SVG diagrams
demos/      interactive demos (one JS file each)
src/        templates, styles, site scripts, demo helper library
build.js    static site generator
SPEC.md     authoring specification and quality bar
site/       generated website (prebuilt)
```

## Syllabus at a glance

1. Foundations: **1** Intro, math and tooling, **2** Neural networks and backpropagation, **3** Optimization, **4** Regularization and training practice
2. Core architectures: **5** CNNs, **6** Modern vision, transfer learning, detection, segmentation, **7** RNNs, LSTMs and attention, **8** Embeddings, representation learning and midterm review
3. Attention and pretrained models: **9** Transformers, **10** LLMs, **11** Vision Transformers, multimodal and self-supervised learning, **12** Autoencoders, VAEs, GANs
4. Advanced topics: **13** Diffusion, flows, autoregressive models, **14** Graph neural networks and geometric deep learning, **15** Deep reinforcement learning, **16** Efficient and scalable deep learning, **17** Trustworthy deep learning and theory, **18** Final projects, review and practice final exam

## Notes on accuracy

The material was written carefully and its numbers spot-checked against published sources, and
all NumPy code that can run offline was executed. PyTorch, Hugging Face, PyTorch Geometric and
Gymnasium examples were syntax-checked but not run in the authoring environment, and reading
lists should be verified against the original papers before being cited. Please open an issue
or pull request for any error you find. See `CONTRIBUTING.md`.

## License

Text and figures: CC BY 4.0 (`LICENSE-CONTENT.md`). Code: MIT (`LICENSE`).
