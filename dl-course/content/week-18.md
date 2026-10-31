---
title: "Final Projects, Course Review, and Final Exam"
phase: Advanced Topics
summary: The capstone week. A guide to giving a 12-minute project talk and writing a conference-format report, with honest reporting of baselines, ablations, seeds, error bars and failure cases; a reproducibility checklist and grading rubric; a cumulative review of Weeks 1 to 17 with key equations, a decision guide and a debugging checklist; a 43-question practice final exam with solutions; and next steps (reading papers, reading lists, reproducing papers, open source, careers).
---

:::key Learning objectives
By the end of this week you should be able to:
- Structure and deliver a 12-minute project talk, and design slides that make exactly one claim each.
- Report experimental results honestly: strong baselines, ablations, multiple seeds, confidence intervals, and failure cases.
- Write a 6 to 8 page report in conference format, section by section, and check it against a rubric before you submit.
- Make a project reproducible: seeds, configs, environment, data versioning, experiment tracking and a model card.
- State the key idea and the key equations of every week of the course, and connect them into one picture.
- Choose an architecture, optimizer, regularizer and generative model for a new problem, and debug a training run systematically.
- Solve a cumulative exam that mixes derivations, conceptual questions and code, with emphasis on Weeks 9 to 17.
- Read a research paper efficiently, reproduce one, and plan what to learn next.
:::

This is the last week and it is a different kind of week. There is no new architecture and no new loss function. Instead you will do three things: show your own work to others, consolidate eighteen weeks of material into a form you can retrieve under time pressure, and decide where to go next.

The week has two sessions plus the exam, following the syllabus. Session 1 is the project presentations (12 minutes plus questions per team). Session 2 is the cumulative review and a discussion of next steps. The final exam is cumulative and emphasizes Weeks 9 to 17. The final report (6 to 8 pages in conference format) and the code repository with a reproducibility README are due this week.

The page follows the same order. Part I is the project guide. Part II is the course review. Part III is a full practice exam. Part IV is about what comes after the course. Skim Part I now if your talk is soon; start Part II and III at least a week before the exam, and do the practice exam under timed conditions.

:::note How to use this page
Reading a review is a weak way to learn; *retrieving* is a strong one. For every week summary in Part II, close the page and write the key equations from memory, then check. For the practice exam, cover the solutions, write your answers on paper, and only then open the solution. Whenever you get something wrong, follow the link to the relevant week and re-read that section, not the whole chapter.
:::

{{figure: week18-concept-map | The whole course on one page. Foundations (blue) supply the loss, the gradient and the optimizer. Architectures (green, purple) are different inductive biases on top of them; generative models, graphs, RL, efficiency and trust (amber, red) apply and stress-test the same machinery. Week 18 (accent) is the scientific practice that wraps around everything.}}

**Part I: the final project**

## Project types and what "good" looks like {#project-types}

Your project is 30% of the course grade (proposal 5%, progress report 5%, final report and code 15%, presentation 5%). The milestones are in the figure below; most of the work should already exist as a running baseline since Week 16.

{{figure: week18-milestones | The project timeline. Each milestone tested a different skill: scoping (proposal), feedback (Week 12), evidence that the baseline works (progress report), and communication (talk and report).}}

Projects that worked well in practice fall into five families. None is better than another; what matters is that the question is sharp and the evidence matches the claim.

| Type | Core question | Typical deliverable | Main risk |
| --- | --- | --- | --- |
| Application | Can deep learning solve problem X on data D better than a sensible baseline? | A tuned pipeline, error analysis, comparison with a classical method | Weak baselines, leakage, no error analysis |
| Method study | Does modification M improve model Z, and why? | Controlled ablations, multiple seeds, a mechanism hypothesis | Gains within seed noise, untuned baseline |
| Reproduction | Do the claims of paper P hold? | Reimplementation, comparison table with the paper, discrepancy analysis | Silent differences in data or evaluation |
| Analysis or interpretability | What does a trained model know or fail at? | Probes, attributions, stress tests, calibration, robustness curves | Over-interpreting a single visualization |
| Systems and efficiency | How fast, small or cheap can we make model Z at a given quality? | Latency, memory and accuracy trade-off curves, profiling | Measuring the wrong thing, ignoring warm-up and batch size |

A useful self-test: finish the sentence "We show that ___ , by measuring ___ , against ___ ." If you cannot fill in all three blanks, the project is not yet a project. The first blank is the claim, the second is the metric and protocol, the third is the baseline. A negative result is fine if it is a *well-supported* negative result: "recipe B does not beat recipe A beyond seed noise at this scale" is a legitimate finding, provided the evidence is strong enough to rule out the effect you looked for.

:::tip Scope for the time you have
A narrow project executed rigorously beats an ambitious one executed sloppily. Prefer small datasets and models that you can train in minutes, because they let you run five seeds and a full ablation grid. A result that came from ten runs on CIFAR-10 is worth more than a result from one run on a model you could barely afford to train once.
:::

## The 12-minute talk {#talk}

Twelve minutes is short. The audience has not read your report, is mixed in background, and will remember about one thing. Plan the talk around that one thing.

{{figure: week18-talk-timeline | A suggested allocation of 12 minutes. Spend the largest share on results, and keep about eight to ten slides. Put anything the audience might ask about, but that does not fit, in backup slides.}}

**Problem (2 minutes).** Say what the task is, give a concrete input and output example, and say why a reader should care. If the audience cannot picture an example input within 30 seconds, you have lost them. End the section with your claim in one sentence.

**Approach (2 minutes).** Describe the baseline and what you changed, with one diagram. Put the architecture and training setup in a single line ("ResNet-18, SGD with momentum, 30 epochs, cosine schedule"). Mention what you did *not* build yourself (pretrained weights, a library), since honesty about provenance is part of the grade.

**Main results (4 minutes).** This is the core. Each slide should state one finding as a headline sentence, show one figure or table, and show uncertainty. Start with the baseline. Show your method next to it, under the same budget and tuning effort. The comparison must be fair and say so.

**Analysis (2 minutes).** An ablation that isolates which part of your method matters, and at least one failure case: examples the model gets wrong, a condition under which the gain vanishes, or a limit of the evaluation. Audiences trust speakers who show failures.

**Wrap-up (2 minutes).** One sentence of takeaway, the limitations, and what you would do with more time. Do not end with "Thank you, questions?" on an empty slide; leave the summary or headline result visible during Q&A.

:::tip Rehearse out loud, with a timer
Almost every first rehearsal is 18 to 20 minutes. Cut slides rather than speaking faster. Rehearse once for time, once for transitions (every slide should end by making the next one inevitable), and once in front of a friend who has not seen the project. Budget for questions: be ready to answer "what is your baseline?", "how many seeds?", "what happens if you remove X?" and "why did you not try Y?".
:::

### Slide anatomy {#slide-anatomy}

{{figure: week18-slide-anatomy | A results slide has five parts: a headline that states the claim, one labeled figure, visible uncertainty, a short setup and takeaway, and a source line. Compare this to a slide titled just "Results".}}

Rules that raise the quality of almost any technical slide:

1. **The title is the claim.** "Cosine decay beats step decay by 0.8 points" is a title; "Learning-rate schedules" is a topic.
2. **One idea per slide.** If a slide needs two minutes of talking, it is two slides.
3. **Figures over tables over text.** Use a table only when exact numbers matter; use at most about 30 words of body text on a slide.
4. **Make plots readable from the back of the room:** font size at least 14 pt, axis labels with units, a legend or direct labels, colors that survive grayscale printing and color-vision deficiency.
5. **Same axes when comparing.** Use the same y-range across panels you want compared, start bar charts at zero (line charts need not), and never truncate an axis to inflate a difference.
6. **Do not read the slide.** The slide holds the evidence; your voice holds the explanation.
7. **Demo with a fallback.** If you run a live demo, record a short video as a backup. Live demos fail at the worst moment.

## Presenting results honestly {#honest-results}

Most of this course has been about *building* models. The work of reporting them is about not fooling yourself, which is harder than it sounds, because training runs are noisy, the hyperparameter search space is huge, and you are biased toward the outcome you hoped for. Five habits protect you.

### Baselines {#baselines}

A result means nothing without something to compare to. Always include (i) a trivial baseline (majority class, last-value forecast, a linear model or logistic regression on features), (ii) a *strong, well-tuned* standard baseline, and (iii) your method. Apply the same tuning budget to all. A very common failure is to tune the new method for a week and run the baseline once with default settings; the gap you then report is mostly the difference in effort. Say in the report how many configurations were tried for each method.

### Ablations {#ablations}

If your method has components A, B and C, an ablation removes one at a time (and sometimes adds one at a time to the baseline) and measures the change. It answers "which part is responsible?" and it protects against the case where the gain comes from an unrelated detail, such as a longer training schedule or a larger batch. Keep everything else fixed, including the number of parameters or the compute budget when those matter. If a component does not help, say so: that is information.

### Seeds and error bars {#seeds}

A neural network's final accuracy depends on the random seed through the initialization, data order, dropout masks and augmentation. The standard deviation across seeds is often 0.2 to 1 point on small benchmarks, comparable to the gains reported by many method papers. So a single run cannot tell a real improvement from luck.

The estimator. Suppose you train $n$ models with different seeds and obtain final metrics $x_1,\dots,x_n$. The sample mean and the (unbiased) sample variance are

$$
\bar x=\frac1n\sum_{i=1}^n x_i,\qquad s^2=\frac{1}{n-1}\sum_{i=1}^n (x_i-\bar x)^2 .
$$

The standard error of the mean is $s/\sqrt n$, and a $95\%$ confidence interval for the *true mean performance of the training recipe* is

$$
\bar x \pm t_{n-1,\,0.975}\,\frac{s}{\sqrt n},
$$

where $t_{n-1,0.975}$ is the 97.5th percentile of Student's $t$ distribution with $n-1$ degrees of freedom (for $n=3$ it is $4.30$, for $n=5$ it is $2.78$, for $n=10$ it is $2.26$, and it tends to $1.96$). The $t$ multiplier matters precisely when $n$ is small, which is the usual case. Note two common confusions: the *standard deviation* $s$ describes how much a single run varies, while the *standard error* $s/\sqrt n$ describes how well you know the mean; and error bars must say which they are.

For comparing two methods trained with $n$ seeds each, a simple and standard test is Welch's $t$ test on the final metrics, with statistic

$$
t=\frac{\bar x_B-\bar x_A}{\sqrt{s_A^2/n_A+s_B^2/n_B}} .
$$

If the models are evaluated on the same test examples you can also bootstrap the test set to capture *test-set sampling* noise, which is a second, separate source of uncertainty. A test set of $N$ examples with accuracy $p$ has a binomial standard error $\sqrt{p(1-p)/N}$: for $p=0.9$ and $N=10{,}000$ this is about $0.3$ points, so differences below that are not distinguishable even with perfectly deterministic training.

Use the explorer below to feel how this works. The gap is the true difference between two training recipes; the standard deviation is the run-to-run noise.

{{demo: week18-seed-explorer | Why single-seed comparisons mislead | 560}}

The next chart quantifies one number from that explorer, computed by simulation (illustrative setting: true gap 0.3 points, per-seed standard deviation 0.5 points, 20,000 simulated experiments for each $n$). It shows how often "B's mean beats A's mean" when B really is better.

```chart
{
  "type": "line",
  "caption": "Probability that the better recipe (true gap 0.3 points, seed std 0.5) also has the higher observed mean, as a function of the number of seeds per recipe. Computed by simulation with NumPy; the data are illustrative of the effect, not of any specific benchmark.",
  "height": 320,
  "data": {
    "labels": [1, 2, 3, 5, 10, 20],
    "datasets": [
      {"label": "P(better method looks better)", "data": [0.664, 0.726, 0.766, 0.831, 0.911, 0.970], "borderColor": "var(--c-accent)", "backgroundColor": "var(--c-accent)", "tension": 0.2},
      {"label": "Coin flip", "data": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "borderColor": "var(--c-muted)", "borderDash": [6, 4], "pointRadius": 0}
    ]
  },
  "options": {"scales": {"x": {"title": {"display": true, "text": "seeds per method"}}, "y": {"min": 0.4, "max": 1, "title": {"display": true, "text": "probability"}}}}
}
```

With one seed, a method that is truly better by 0.3 points loses a third of the time. The chart also shows why "run more seeds" has diminishing returns: the interval shrinks like $1/\sqrt n$, so to halve it you need four times as many runs.

```chart
{
  "type": "bar",
  "caption": "Half-width of the 95% confidence interval of the mean (Student t, seed std 0.5 points) as a function of the number of seeds. With 2 seeds the interval is wider than the effects most papers report; with 5 it is about 0.6 points; with 20 about 0.23.",
  "height": 320,
  "data": {
    "labels": ["2", "3", "5", "10", "20"],
    "datasets": [{"label": "CI half-width (points)", "data": [4.49, 1.24, 0.62, 0.36, 0.23], "backgroundColor": "var(--c-blue)"}]
  },
  "options": {"scales": {"x": {"title": {"display": true, "text": "seeds"}}, "y": {"title": {"display": true, "text": "95% CI half-width (points)"}}}}
}
```

:::warning What not to do
Do not report the best of $n$ seeds for your method and the average for the baseline. Do not pick the seed after looking at the test set. Do not tune hyperparameters on the test set (even indirectly, by choosing the checkpoint with the best test accuracy); tune on validation data and evaluate the test set once. If you have to deviate, say so in the report.
:::

### Reporting the full picture {#full-picture}

Report means with an interval, and when $n$ is small also show the individual points. State the number of seeds, the split used to select hyperparameters, and the metric definition. When differences are small, report a *paired* analysis (same seeds, same data order, same test examples) which removes some noise. When your claim concerns more than one dataset or setting, show them all rather than the favorable one.

### Failure cases {#failure-cases}

The last habit is the most neglected. Look at the errors. Sort test examples by loss and look at the worst fifty. Slice the metric by subgroup, input length, class, or difficulty. Check calibration (see [Week 17](week-17.html#uncertainty)). Try a condition outside the training distribution (see [Week 17](week-17.html#shift)). In the talk, show one or two failures and say what they reveal: label noise, an ambiguous class, a shortcut feature, a regime where your method gives no gain. This is what makes the analysis a contribution and not just a number.

## The final report {#report}

The report is 6 to 8 pages (excluding references and appendix) in a standard conference format (the NeurIPS, ICML or ICLR LaTeX styles are all acceptable; use one consistently). Use a single column of text with figures that are legible when printed. Below is a section-by-section template with a checklist for each section.

### Abstract (about 150 words) {#report-abstract}

Four sentences do most of the work: the problem, the approach, the main result with a number, and the main takeaway. Check that: it names the task and dataset; it states a quantitative result with uncertainty; it makes no claim that the paper does not support.

### Introduction (about 1 page) {#report-intro}

Motivate the problem, state what is hard, summarize your approach, and *list your contributions* in two to four bullets. Check that: a reader outside your sub-area understands the task; each contribution is checkable against a section or figure; claims are scoped ("on CIFAR-10 with ResNet-18", not "for vision").

### Related work (about 0.5 to 1 page) {#report-related}

Group prior work by idea, and say how yours differs. Do not list papers; relate them. Check that: the strongest prior method on your task is cited and, ideally, compared; every citation is real and was actually read; text is your own words.

### Method (1 to 1.5 pages) {#report-method}

Define the notation, give the model and loss, and describe the algorithm so that an informed reader could reimplement it. A diagram is usually worth more than a paragraph. Check that: every symbol is defined once; shapes and hyperparameters are stated; any non-obvious design choice has a reason.

### Experimental setup (about 1 page) {#report-setup}

Datasets (with sizes and splits), preprocessing and augmentation, baselines, metrics, hyperparameter search protocol and budget, number of seeds, hardware and training time. Check that: the validation and test roles are explicit; baselines receive comparable tuning; you say which results are your own runs and which are copied from papers.

### Results (1.5 to 2 pages) {#report-results}

Main table or figure with mean and interval, then ablations, then analysis. Check that: the table caption states what the numbers are (mean over how many seeds, which interval); the best result is bolded only when the difference is meaningful; negative results are included.

### Analysis and failure cases (about 0.5 to 1 page) {#report-analysis}

Where does the method fail, and why do you think so? What did the ablations say? Include at least one hypothesis that you tested, and say whether it was supported. Check that: failures are shown, not only described; speculation is marked as such.

### Limitations and conclusion (about 0.5 page) {#report-conclusion}

State limits of the data, scale, evaluation and compute, then the conclusion, then future work. Check that: the conclusion restates the *supported* claims only; ethical and societal considerations are addressed where relevant (data provenance, bias, misuse).

### References and appendix {#report-appendix}

References in one consistent style. The appendix holds the full hyperparameter table, extra seeds, additional plots and the model card. Check that: every reference is complete and verifiable; the appendix is enough to rerun the experiments.

:::note AI assistants and external code
Course policy allows AI coding assistants and external code only with disclosure and citation. Add a short "Acknowledgements and tools" paragraph that says what you used and for what. Anything you submit must be something you understand and can explain in the Q&A.
:::

## Reproducibility {#reproducibility}

A result is *reproducible* if someone else can obtain the same result from your code and data, and *replicable* if they get a consistent result from fresh data and code. For a course project, the goal is the first: a stranger (including you in six months) should be able to run one command and regenerate every table and figure in your report.

{{figure: week18-repro-stack | Five things a run must record so that it can be reproduced: the code version, the configuration, the data version, the software and hardware environment, and the sources of randomness. Store them together with the outputs of every run.}}

### The checklist {#repro-checklist}

- **Seeds.** Set Python, NumPy and PyTorch seeds; seed the DataLoader workers and generator; record the seeds in the run record. Different seeds for different replicates, the *same* seed list for compared methods (paired comparison).
- **Determinism.** Know which flags exist (`torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.benchmark=False`) and what they cost. Exact bitwise reproducibility across GPU types is often impossible; aim for statistical reproducibility, and report seed variance.
- **Configs.** Put all hyperparameters in one config file (YAML or a dataclass), not in scattered constants or command-line habits. Save the resolved config with every run.
- **Environment.** Pin versions (`requirements.txt` with exact versions, or a `conda` lock file or a Docker image). Record Python, PyTorch, CUDA and GPU model.
- **Data versioning.** Record the dataset name, version or checksum, and the exact train, validation and test split (store the indices or a split seed). If you preprocess, script it and store the script. Tools such as DVC or simply a hash in the run record work.
- **Code versioning.** Run from a clean git commit and record the hash; if the tree is dirty, store the diff.
- **Metrics and artifacts.** Log metrics per step or epoch to a file (and optionally to a tracker such as Weights & Biases, MLflow or TensorBoard), save the final checkpoint and the evaluation outputs.
- **README.** State what the repository does, how to install, how to run each experiment (exact commands), expected runtime and hardware, expected results (the numbers in the report), and known issues.
- **Model card.** For any model you release, document intended use, training data, evaluation results (including subgroups), limitations and ethical considerations (Mitchell et al., 2019).

### A minimal tracking harness {#tracking-code}

The code below is a self-contained harness that does most of the checklist in a few dozen lines: it fixes seeds, captures the environment and git state, saves the config, appends metrics as JSON lines, and writes a final summary. It uses only the standard library plus NumPy and (optionally) PyTorch, so you can drop it into any project.

```python:tracking.py
import json, os, platform, random, subprocess, sys, time, hashlib
from dataclasses import dataclass, asdict, field

import numpy as np

try:
    import torch
except ImportError:  # the harness still works without torch
    torch = None


@dataclass
class Config:
    """All hyperparameters live here and are saved with every run."""
    name: str = "baseline"
    seed: int = 0
    lr: float = 3e-4
    batch_size: int = 128
    epochs: int = 10
    weight_decay: float = 0.01
    extra: dict = field(default_factory=dict)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every source of randomness we use."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Pass as worker_init_fn to DataLoader so each worker has a distinct, reproducible seed."""
    base = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(base)
    random.seed(base)


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": run("git", "status", "--porcelain") != ""}


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__}
    if torch is not None:
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return info


class Run:
    """One experiment run: a directory with config, environment, metrics and summary."""

    def __init__(self, cfg: Config, root: str = "runs"):
        self.cfg = cfg
        tag = hashlib.sha1(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:8]
        self.dir = os.path.join(root, f"{cfg.name}-s{cfg.seed}-{tag}")
        os.makedirs(self.dir, exist_ok=True)
        set_seed(cfg.seed)
        meta = {"config": asdict(cfg), "git": git_info(), "env": env_info(),
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self.metrics_file = open(os.path.join(self.dir, "metrics.jsonl"), "a")

    def log(self, step: int, **metrics) -> None:
        self.metrics_file.write(json.dumps({"step": step, **metrics}) + "\n")
        self.metrics_file.flush()

    def finish(self, **summary) -> None:
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        self.metrics_file.close()


if __name__ == "__main__":
    # Smoke test: a fake "training run" whose final metric depends on the seed.
    cfg = Config(name="demo", seed=1)
    run = Run(cfg)
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(cfg.epochs):
        run.log(epoch, train_loss=1.0 / (epoch + 1) + 0.01 * rng.normal())
    run.finish(test_acc=0.9 + 0.005 * rng.normal())
    print("wrote", run.dir)
```

### A seed sweep with confidence intervals {#seed-sweep}

The second snippet turns a list of per-seed results into the numbers that belong in your results table: mean, standard deviation, a $t$-based confidence interval, a paired comparison, and a bootstrap interval for the difference. It needs only NumPy; the $t$ quantile is hard-coded for a few common values to avoid a SciPy dependency, with SciPy used if available.

```python:seed_stats.py
import numpy as np

# 97.5th percentile of Student's t for df = 1..30 (two-sided 95% intervals)
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t975(df: int) -> float:
    return _T975[df - 1] if df <= 30 else 1.96


def summarize(x):
    """Mean, sample std (ddof=1), standard error and 95% CI half-width."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    std = x.std(ddof=1) if n > 1 else float("nan")
    sem = std / np.sqrt(n) if n > 1 else float("nan")
    half = t975(n - 1) * sem if n > 1 else float("nan")
    return dict(n=n, mean=mean, std=std, sem=sem, ci_half=half)


def paired_bootstrap(a, b, n_boot=10_000, seed=0):
    """Bootstrap CI for mean(b - a) when a[i] and b[i] share a seed (paired runs)."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return d.mean(), np.percentile(means, [2.5, 97.5])


if __name__ == "__main__":
    # Final test accuracy (%) of two recipes over the same 5 seeds (made-up numbers for illustration).
    A = [91.2, 90.8, 91.5, 90.9, 91.1]
    B = [91.6, 91.0, 91.9, 91.4, 91.3]
    for name, x in [("A", A), ("B", B)]:
        s = summarize(x)
        print(f"{name}: {s['mean']:.2f} +- {s['ci_half']:.2f} (std {s['std']:.2f}, n={s['n']})")
    gap, (lo, hi) = paired_bootstrap(A, B)
    print(f"paired gap B-A: {gap:.2f}, 95% bootstrap CI [{lo:.2f}, {hi:.2f}]")
    print("claim 'B > A' supported" if lo > 0 else "claim 'B > A' NOT supported at 95%")
```

```text
A: 91.10 +- 0.37 (std 0.30, n=5)
B: 91.44 +- 0.40 (std 0.32, n=5)
paired gap B-A: 0.34, 95% bootstrap CI [0.26, 0.42]
```

Notice what the paired analysis does: the unpaired intervals for A and B overlap, yet the paired differences are consistently positive, because both recipes share the "hard" and "easy" seeds. Pairing is legitimate only if seeds really control the same shared factors (for example the same data order), so say so when you use it. With only five pairs a bootstrap is crude; treat it as a diagnostic and be conservative in the wording.

## Grading rubric {#rubric}

The final report and presentation are graded on the criteria stated in the syllabus. The rubric below makes them concrete (100 points; the report and code carry 15% of the course, the presentation 5%).

| Criterion | Points | Full marks look like |
| --- | --- | --- |
| Problem significance and clarity | 10 | A sharp question, well motivated, scoped to what the evidence can support |
| Technical correctness | 20 | Correct method and implementation; no leakage; sound evaluation protocol |
| Experimental rigor | 25 | Strong, equally tuned baselines; ablations; at least 3 to 5 seeds; intervals; failure cases |
| Analysis and insight | 15 | Explains why results look as they do; tests at least one hypothesis; honest about negatives |
| Clarity of writing and figures | 15 | Conference-style report with the checklist above, readable figures, correct citations |
| Reproducibility | 10 | README with exact commands, pinned environment, configs and seeds, data instructions, model card |
| Presentation | 5 (scaled separately) | On time, one claim per slide, uncertainty visible, handles questions well |

Common deductions: results from a single seed presented as conclusive; baseline weaker than a standard one; tuning on the test set; claims in the abstract that the experiments do not show; missing citation for borrowed code; a repository that cannot be run from the README.

## Course review: how to study {#review-intro}

**Part II: cumulative review of Weeks 1 to 17**

The final exam is cumulative with emphasis on Weeks 9 to 17. The midterm review in [Week 8](week-08.html#midterm-review) covers the first seven weeks in more depth; here each week gets a compact "must know" summary with its key equations and the one idea you should be able to explain to a colleague. Start by rating yourself.

{{demo: week18-readiness-radar | Exam readiness self-check | 720}}

Use the result as a plan: spend your first study hours on the weeks with the highest priority score, then do the practice exam in Part III, then revisit what you missed.

## Week-by-week must-know summaries {#weekly-summaries}

Each entry is short on purpose. If any equation looks unfamiliar, follow the link.

### Week 1: Foundations {#rev-w1}

[Week 1](week-01.html) set up the language: a model $f_\theta$, a loss $\mathcal{L}$, data, and gradient descent. Know:

- Supervised learning as empirical risk minimization, and maximum likelihood: minimizing cross-entropy equals maximizing the log-likelihood; mean squared error is Gaussian MLE.
- Gradient descent: $\theta_{t+1}=\theta_t-\eta\nabla_\theta\mathcal{L}(\theta_t)$, with the minibatch gradient an unbiased estimate of the full gradient.
- Information theory: $H(p)=-\sum p\log p$, $\KL(p\|q)=\sum p\log(p/q)\ge 0$, and cross-entropy $H(p,q)=H(p)+\KL(p\|q)$.
- Tensor shapes and broadcasting, and the PyTorch loop: forward, loss, `zero_grad`, `backward`, `step`.

### Week 2: Neural networks and backpropagation {#rev-w2}

[Week 2](week-02.html): an MLP layer is $\boldsymbol{h}^{(l)}=\phi(\mathbf{W}^{(l)}\boldsymbol{h}^{(l-1)}+\boldsymbol{b}^{(l)})$. Know:

- Softmax cross-entropy gradient w.r.t. the logits: $\partial\mathcal{L}/\partial z_k=p_k-y_k$.
- Backprop: $\delta^{(l)}=(\mathbf{W}^{(l+1)\top}\delta^{(l+1)})\odot\phi'(a^{(l)})$, $\partial\mathcal{L}/\partial\mathbf{W}^{(l)}=\delta^{(l)}\boldsymbol{h}^{(l-1)\top}$. It is reverse-mode autodiff: cost about the same as one forward pass.
- Universal approximation: one hidden layer suffices in principle; depth gives efficiency. Why nonlinearity is needed, and the properties of ReLU, GELU, sigmoid and tanh.

### Week 3: Optimization {#rev-w3}

[Week 3](week-03.html): momentum $v\leftarrow\beta v+g,\ \theta\leftarrow\theta-\eta v$; Adam with $m\leftarrow\beta_1 m+(1-\beta_1)g$, $v\leftarrow\beta_2 v+(1-\beta_2)g^2$, bias correction $\hat m=m/(1-\beta_1^t)$, update $\theta\leftarrow\theta-\eta\hat m/(\sqrt{\hat v}+\epsilon)$. AdamW decouples weight decay from the adaptive scaling. Know:

- Condition number $\kappa$ and the GD rate $(\kappa-1)/(\kappa+1)$ per step on a quadratic; why momentum helps.
- Initialization: Xavier $\mathrm{Var}(w)=2/(n_{in}+n_{out})$, He $2/n_{in}$; goal is to keep activation and gradient variance constant across layers.
- BatchNorm $\hat x=(x-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$ with learned $\gamma,\beta$; train and eval modes differ; LayerNorm normalizes across features.
- Warmup plus cosine decay as the default schedule.

### Week 4: Regularization and generalization {#rev-w4}

[Week 4](week-04.html): the bias-variance decomposition $\E[(y-\hat f)^2]=\text{bias}^2+\text{variance}+\sigma^2$. L2 is a Gaussian prior (MAP), L1 a Laplace prior (sparsity). Inverted dropout scales by $1/(1-p)$ at train time. Know double descent, label smoothing, early stopping, augmentation, hyperparameter search (random beats grid), and the train/validation/test discipline.

### Week 5: Convolutional networks {#rev-w5}

[Week 5](week-05.html): output size $\lfloor(n+2p-d(k-1)-1)/s\rfloor+1$; parameters $C_{out}(C_{in}k^2+1)$; translation equivariance of convolution, approximate invariance after pooling; receptive field growth; depthwise-separable cost ratio $\approx 1/C_{out}+1/k^2$.

### Week 6: Modern vision {#rev-w6}

[Week 6](week-06.html): the residual block $\boldsymbol{x}_{l+1}=\boldsymbol{x}_l+F(\boldsymbol{x}_l)$ gives $\partial\boldsymbol{x}_L/\partial\boldsymbol{x}_l=\mathbf{I}+\partial(\sum F)/\partial\boldsymbol{x}_l$, an identity path for gradients. Know transfer learning and fine-tuning, IoU, non-maximum suppression, mAP, and the roles of encoder-decoder skip connections in U-Net segmentation.

### Week 7: Sequence models {#rev-w7}

[Week 7](week-07.html): RNN $\boldsymbol{h}_t=\tanh(\mathbf{W}_h\boldsymbol{h}_{t-1}+\mathbf{W}_x\boldsymbol{x}_t+\boldsymbol{b})$; backprop through time multiplies Jacobians $\prod_t\partial\boldsymbol{h}_t/\partial\boldsymbol{h}_{t-1}$, which vanishes or explodes; clip gradients for the latter. LSTM: cell $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$ has an additive path. Seq2seq with attention removes the fixed-vector bottleneck: $\alpha_{ti}=\softmax_i(e_{ti})$, context $=\sum_i\alpha_{ti}\boldsymbol{h}_i$.

### Week 8: Representations {#rev-w8}

[Week 8](week-08.html): skip-gram with negative sampling maximizes $\log\sigma(\boldsymbol{u}_o^\top\boldsymbol{v}_c)+\sum_{k}\log\sigma(-\boldsymbol{u}_k^\top\boldsymbol{v}_c)$ (and implicitly factorizes a shifted PMI matrix). BPE builds subwords by merging frequent pairs. InfoNCE: $-\log\frac{\exp(s_+/\tau)}{\sum_j\exp(s_j/\tau)}$ is a lower bound on mutual information. Embeddings encode data biases.

### Week 9: Transformers {#rev-w9}

[Week 9](week-09.html) is the most important architecture week. Know by heart:

$$
\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})=\softmax\!\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\Big)\mathbf{V},\qquad \mathrm{head}_i=\mathrm{Attention}(\mathbf{X}\mathbf{W}^Q_i,\mathbf{X}\mathbf{W}^K_i,\mathbf{X}\mathbf{W}^V_i).
$$

- The $1/\sqrt{d_k}$ factor keeps score variance at 1 when query and key entries have unit variance, so softmax does not saturate.
- Cost: attention is $O(T^2 d)$ time and $O(T^2)$ memory per head; the FFN is $O(T d^2)$.
- Parameters per block: about $4d^2$ for attention and $8d^2$ for the FFN (with hidden width $4d$), so about $12d^2$ per layer; the total is about $12Ld^2$ plus embeddings.
- Causal mask for decoders; pre-LN is more stable than post-LN; positional information (sinusoidal, learned, RoPE, ALiBi); encoder-only (BERT), decoder-only (GPT) and encoder-decoder (T5).

### Week 10: LLMs, pretraining and alignment {#rev-w10}

[Week 10](week-10.html): next-token loss $-\sum_t\log p_\theta(x_t\mid x_{<t})$. Scaling laws: loss falls as a power law in parameters, data and compute; compute-optimal training (Chinchilla) balances parameters and tokens, roughly 20 tokens per parameter. Training compute is about $6ND$ FLOPs for $N$ parameters and $D$ tokens. LoRA: $\mathbf{W}+\tfrac{\alpha}{r}\mathbf{B}\mathbf{A}$ with rank $r\ll d$. RLHF: reward model from pairwise preferences with the Bradley-Terry loss $-\log\sigma(r(x,y_w)-r(x,y_l))$, then policy optimization with a KL penalty; DPO optimizes $-\log\sigma\big(\beta[\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}]\big)$ directly. Decoding: greedy, beam, temperature, top-$k$, top-$p$. RAG adds retrieved text to the prompt.

### Week 11: Vision transformers and multimodal models {#rev-w11}

[Week 11](week-11.html): ViT splits an image into $P\times P$ patches, so $N=HW/P^2$ tokens, linearly embeds them and adds a class token and positional embeddings; weaker inductive bias than CNNs means it needs more data or strong augmentation and distillation (DeiT). Swin restricts attention to shifted windows for linear cost. Self-supervision: contrastive (SimCLR, MoCo), self-distillation (DINO), masked image modeling (MAE). CLIP trains image and text encoders with a symmetric InfoNCE over a batch similarity matrix and enables zero-shot classification by prompting.

### Week 12: VAEs and GANs {#rev-w12}

[Week 12](week-12.html): the ELBO $\log p_\theta(x)\ge\E_{q_\phi(z|x)}[\log p_\theta(x|z)]-\KL(q_\phi(z|x)\|p(z))$; the reparameterization trick $z=\mu+\sigma\odot\epsilon$ gives low-variance gradients through sampling. The GAN game $\min_G\max_D\E_x[\log D(x)]+\E_z[\log(1-D(G(z)))]$; at the optimal discriminator the objective is $2\,\mathrm{JSD}(p_{data}\|p_g)-\log 4$. Wasserstein GAN replaces JSD by the earth-mover distance with a Lipschitz critic (clipping or gradient penalty). Know mode collapse, FID and its limits.

### Week 13: Diffusion and other generative models {#rev-w13}

[Week 13](week-13.html): forward process $q(x_t|x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)\mathbf{I})$, i.e. $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$. The simplified loss is $\E\|\epsilon-\epsilon_\theta(x_t,t)\|^2$. The noise prediction is related to the score by $\nabla_{x_t}\log q(x_t)\approx-\epsilon_\theta/\sqrt{1-\bar\alpha_t}$. DDIM gives a deterministic sampler in far fewer steps; classifier-free guidance uses $\tilde\epsilon=(1+w)\epsilon_\theta(x_t,c)-w\,\epsilon_\theta(x_t,\varnothing)$; latent diffusion runs in an autoencoder's latent space. Compare against VAEs (blurry, fast), GANs (sharp, unstable, no likelihood), flows (exact likelihood, restricted architectures), and autoregressive models (exact likelihood, slow sampling).

### Week 14: Graph neural networks {#rev-w14}

[Week 14](week-14.html): message passing $\boldsymbol{h}_v^{(k+1)}=\phi\big(\boldsymbol{h}_v^{(k)},\bigoplus_{u\in\mathcal{N}(v)}\psi(\boldsymbol{h}_u^{(k)},\boldsymbol{h}_v^{(k)})\big)$ with a permutation-invariant aggregator. GCN: $\mathbf{H}'=\sigma(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{H}\mathbf{W})$ with $\tilde{\mathbf{A}}=\mathbf{A}+\mathbf{I}$. GAT learns attention over neighbours; GIN uses sum aggregation and is as expressive as the 1-WL test. Over-smoothing (features converge as depth grows) and over-squashing (long-range information compressed through bottlenecks). Permutation invariance and equivariance as the design principle of geometric deep learning.

### Week 15: Deep reinforcement learning {#rev-w15}

[Week 15](week-15.html): the Bellman equation $Q^\pi(s,a)=\E[r+\gamma Q^\pi(s',a')]$ and the optimality version with a max. DQN minimizes $\big(r+\gamma\max_{a'}Q_{\bar\theta}(s',a')-Q_\theta(s,a)\big)^2$ using a replay buffer and a target network to tame the deadly triad (function approximation, bootstrapping, off-policy). Policy gradient: $\nabla_\theta J=\E[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)\,\hat A_t]$, with a baseline for variance reduction. GAE $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$. PPO clips the ratio: $\min(r_t\hat A_t,\ \mathrm{clip}(r_t,1\pm\epsilon)\hat A_t)$. SAC adds an entropy bonus. Offline RL fights extrapolation error.

### Week 16: Efficiency and scale {#rev-w16}

[Week 16](week-16.html): arithmetic intensity $I=\text{FLOPs}/\text{bytes}$ and the roofline: attainable throughput $=\min(\text{peak FLOPs},\ I\times\text{bandwidth})$. Mixed precision (bf16 for range, fp16 with loss scaling); gradient accumulation to emulate large batches; checkpointing trades compute for memory (about one extra forward pass); DDP all-reduces gradients; ZeRO and FSDP shard optimizer state, gradients and parameters. For Adam in mixed precision the per-parameter memory is about 16 bytes (2 fp16 weight, 2 fp16 gradient, 4+4+4 fp32 master weight and two moments). FlashAttention computes exact attention in tiles to avoid materializing the $T\times T$ matrix. KV caching makes decoding $O(T)$ per token; quantization, pruning, distillation and MoE trade quality, memory and compute.

### Week 17: Trustworthy deep learning and theory {#rev-w17}

[Week 17](week-17.html): FGSM $x'=x+\epsilon\,\mathrm{sign}(\nabla_x\mathcal{L})$ and PGD (iterated, projected); adversarial training solves $\min_\theta\E\max_{\|\delta\|\le\epsilon}\mathcal{L}$. Distribution shift (covariate, label, concept), max-softmax and energy OOD scores. Calibration: ECE $=\sum_b\frac{|B_b|}{n}|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)|$, temperature scaling $\softmax(z/T)$, split conformal prediction with finite-sample coverage $\ge1-\alpha$. Grad-CAM, Integrated Gradients (completeness: attributions sum to $f(x)-f(x')$), DP-SGD (clip per-example gradients, add Gaussian noise). Demographic parity versus equalized odds. Theory: randomization tests, implicit regularization, NTK, lottery tickets, grokking.

## Threads that connect the weeks {#threads}

Exam questions in the last third of the course often combine weeks. Five connecting ideas to keep in mind:

1. **Everything is maximum likelihood or a close relative.** Cross-entropy (classification, language modeling), MSE (regression, diffusion noise prediction), ELBO (VAEs and diffusion), the policy-gradient objective and InfoNCE are all expected log-probability objectives or bounds on them.
2. **Inductive bias versus data.** CNNs, RNNs, GNNs and Transformers hard-wire different symmetries (translation, time, permutation, none). Less bias needs more data or pretraining; more bias generalizes better from little data but can be limiting.
3. **Gradient flow.** Residual connections, normalization, initialization, gating and gradient clipping all keep products of Jacobians well behaved. Many "model does not train" problems are gradient-flow problems.
4. **Resources.** Memory and bandwidth, not FLOPs, are frequently the real limits (roofline, KV cache, FlashAttention). Scaling laws tell you how to split a compute budget.
5. **Evaluation under uncertainty.** Seeds, calibration, shift and subgroup analysis answer the question "does this result and this model deserve trust?".

## Decision guide: what to use when {#decision-guide}

These are defaults, not laws; they encode what the field has learned to try first.

{{figure: week18-decision-tree | A starting-point guide for the model family by data type. The first model you try should be the simplest strong baseline for that data; change one thing at a time.}}

### Which architecture {#decide-architecture}

| Situation | Default | Why, and when to deviate |
| --- | --- | --- |
| Images, small data | Pretrained CNN (ResNet, ConvNeXt, EfficientNet), fine-tune | Strong locality prior, cheap; deviate to ViT when pretraining data is large |
| Images, large data or multimodal | ViT or hierarchical transformer; CLIP-style encoders | Scales better; patch size controls cost |
| Dense prediction | U-Net or a transformer decoder; DETR or Mask R-CNN for detection | Skip connections keep resolution |
| Text understanding | Pretrained encoder (BERT-like) fine-tuned, or an LLM with prompting | Decoder-only models are often best at scale |
| Text generation, assistants | Decoder-only Transformer, instruction tuning, RLHF or DPO | LoRA or QLoRA if GPU memory is limited |
| Long sequences | Efficient attention (FlashAttention, windowed, state space models) | Quadratic cost dominates beyond a few thousand tokens |
| Time series | Start with a linear or gradient-boosting baseline, then a TCN or Transformer | Deep models do not always win |
| Graphs | GCN or GraphSAGE baseline, GAT or GIN when edges or expressivity matter | Watch over-smoothing beyond 3 or 4 layers |
| Tabular | Gradient boosting first; MLP with embeddings as a comparison | Deep nets win mostly with large data or mixed modalities |
| Decision making | PPO for on-policy, SAC or TD3 for continuous control, DQN variants for discrete | Offline data: conservative methods; or consider supervised imitation first |

### Which optimizer and schedule {#decide-optimizer}

- **Default:** AdamW (learning rate $10^{-4}$ to $10^{-3}$ for small nets, $10^{-5}$ to $10^{-4}$ for fine-tuning pretrained Transformers), weight decay 0.01 to 0.1, warmup then cosine decay, gradient clipping at 1.0 for Transformers and RNNs.
- **SGD with momentum** (0.9, learning rate around 0.1 with batch size 256, step or cosine decay) still tends to generalize slightly better for CNNs on vision benchmarks.
- Scale the learning rate with batch size (linear scaling with warmup is the common rule of thumb); use a learning-rate range test to find the right order of magnitude.
- Mixed precision (bf16 where available) is the default on modern GPUs.

### Which regularizer {#decide-regularizer}

Start with **weight decay, data augmentation and early stopping**. Add **dropout** in Transformers and large MLPs (rate 0.1), not usually in CNNs with BatchNorm. Use **label smoothing** (0.1) for classification, **mixup or CutMix** for vision, **stochastic depth** for very deep networks, and **more data or pretraining** whenever you can get it, which beats every trick. If the train-validation gap is small and both are poor, you need a bigger model or longer training, not more regularization.

### Which generative model {#decide-generative}

| Need | Choice | Reason |
| --- | --- | --- |
| Highest-quality images, text-conditioned | Latent diffusion with guidance | Stable training, flexible conditioning |
| Real-time generation | GAN or distilled or consistency-style samplers | One or few network evaluations |
| Representation and compression, smooth latent space | VAE | Encoder, ELBO, interpretable latent |
| Exact likelihood, density estimation | Autoregressive model or normalizing flow | Tractable $\log p(x)$ |
| Text, code, discrete sequences | Autoregressive Transformer | Natural factorization, scales well |

## Debugging checklist {#debugging}

When a run fails, work from simple to complex. Almost all failures are bugs in data or code, not in the idea.

1. **Overfit one batch.** Take 8 to 32 examples and train to near-zero loss. If you cannot, there is a bug in the model, loss, labels or optimizer step.
2. **Check the initial loss.** For $K$-class classification with random initialization it should be about $\ln K$ (for example $2.30$ for 10 classes). Far off means a wrong final layer scale, a missing softmax or wrong labels.
3. **Inspect data.** Visualize inputs after preprocessing and augmentation, with their labels. Check the label range, class balance, train-test leakage, and normalization statistics.
4. **Check shapes and dimensions.** Silent broadcasting bugs are the most common ones (a loss of shape $(B,1)$ minus $(B,)$ gives $(B,B)$).
5. **Check modes and devices.** `model.train()` versus `model.eval()` (dropout, BatchNorm), `torch.no_grad()` at inference, gradients actually non-zero and not `NaN`.
6. **Learning rate first.** Sweep it over several orders of magnitude; most "it does not learn" problems are a bad learning rate. Plot the loss curve per step.
7. **Track gradient and update norms,** and the ratio of update to weight norm (about $10^{-3}$ is a healthy order of magnitude as a rule of thumb).
8. **Loss goes to NaN:** lower the learning rate, add warmup and gradient clipping, check for division by zero or $\log 0$, use a numerically stable loss (`cross_entropy` on logits, not `log(softmax)`).
9. **Train good, validation bad:** overfitting (augmentation, weight decay, more data) or a train-eval mismatch such as BatchNorm statistics or leakage in the split.
10. **Both bad:** underfitting, a bug, or an optimization problem. Increase capacity, train longer, fix the schedule.
11. **Compare against a trivial baseline** and a known-good reference implementation on the same data.
12. **Change one thing at a time,** keep a log, and make runs reproducible so you can bisect.

```chart
{
  "type": "radar",
  "caption": "Illustrative exam-weight profile: how many of the 43 practice questions below touch each course phase (a question can count toward more than one phase). The distribution mirrors the exam emphasis on Weeks 9 to 17.",
  "height": 380,
  "data": {
    "labels": ["Foundations W1-4", "Core architectures W5-8", "Transformers and LLMs W9-10", "Vision and multimodal W11", "Generative W12-13", "Graphs W14", "RL W15", "Efficiency W16", "Trust and theory W17"],
    "datasets": [{"label": "Questions", "data": [5, 4, 8, 3, 6, 3, 4, 4, 6], "borderColor": "var(--c-accent)", "backgroundColor": "rgba(80,120,220,0.2)"}]
  }
}
```

## Practice final exam {#practice-exam}

**Part III: a cumulative practice final**

Forty-three questions, mixing derivations, conceptual questions, numeric calculations and coding. About two thirds touch Weeks 9 to 17, matching the exam emphasis. A realistic target is two to three hours for the whole set; the real exam will be shorter. Work on paper first, then open the solution. Questions are grouped by topic and each is tagged with its week.

:::note Marking yourself
For each question award yourself full credit only if your reasoning, not just your final answer, matches the solution. If you could only do it after reading the solution, count it as a miss and revisit the linked week.
:::

### Section A: Foundations (Weeks 1 to 4) {#exam-a}

:::exercise Question 1 (Week 2): softmax cross-entropy gradient
For logits $\boldsymbol{z}\in\R^K$, $\boldsymbol{p}=\softmax(\boldsymbol{z})$ and a one-hot label $\boldsymbol{y}$, derive $\partial\mathcal{L}/\partial z_k$ for $\mathcal{L}=-\sum_j y_j\log p_j$.
:::details Solution
$\mathcal{L}=-z_c+\log\sum_j e^{z_j}$ where $c$ is the true class. Then $\partial\mathcal{L}/\partial z_k=-\mathbb{1}[k=c]+e^{z_k}/\sum_j e^{z_j}=p_k-y_k$. The gradient is the prediction minus the target; it is bounded in $[-1,1]$ and vanishes only when the prediction matches the label, which is why softmax with cross-entropy does not suffer the saturation of sigmoid with squared error.
:::
:::

:::exercise Question 2 (Weeks 2 and 4): initial loss and overfitting one batch
You train a 10-class classifier. At step 0 the loss is 14.7. After 500 steps on the full dataset it is 2.30 and stays there. (a) What would you expect the step-0 loss to be and what does 14.7 suggest? (b) What does a plateau at 2.30 mean? (c) Name the first experiment you run.
:::details Solution
(a) A randomly initialized network with small outputs predicts nearly uniform probabilities, so the loss is $\ln 10\approx2.30$. A value of 14.7 means large logits (scale too big, bad initialization, or an unnormalized input with huge magnitude) making the model confidently wrong. (b) 2.30 means the model has collapsed to predicting the uniform distribution (or the class prior): either it is not learning at all (learning rate too small or too large and the network died, labels shuffled or unrelated to inputs, dead ReLUs) or an input pipeline bug. (c) Overfit a single small batch: it should reach near-zero loss within a few hundred steps; if not, the bug is in the model, loss or optimizer rather than in the data volume.
:::
:::

:::exercise Question 3 (Week 3): the first Adam step
Starting with $m_0=v_0=0$, show that the first Adam update with bias correction is $\eta\,g/(|g|+\epsilon)$ per coordinate, and say what that implies about the scale of early updates. What goes wrong if you skip the bias correction with $\beta_2=0.999$?
:::details Solution
$m_1=(1-\beta_1)g$ and $v_1=(1-\beta_2)g^2$. The corrections divide by $1-\beta_1$ and $1-\beta_2$: $\hat m_1=g$, $\hat v_1=g^2$. The update is $\eta\,g/(\sqrt{g^2}+\epsilon)=\eta\,g/(|g|+\epsilon)\approx\eta\,\mathrm{sign}(g)$: every coordinate moves by about $\eta$ regardless of gradient scale. Without correction, $m_1=0.1g$ and $\sqrt{v_1}=\sqrt{0.001}\,|g|\approx0.032|g|$, so the step would be about $3.2\,\eta$, a first step 3 times larger than intended; the bias is worse for $\beta_2$ because $v$ starts at zero and takes about 1000 steps to be unbiased. This, with the sign-like first steps, is why warmup is recommended.
:::
:::

:::exercise Question 4 (Week 4): weight decay as a prior
Show that minimizing $\sum_i(y_i-\boldsymbol{w}^\top\boldsymbol{x}_i)^2+\lambda\norm{\boldsymbol{w}}^2$ is the MAP estimate under Gaussian noise with variance $\sigma^2$ and a prior $\boldsymbol{w}\sim\mathcal{N}(0,\tau^2\mathbf{I})$, and identify $\lambda$. What prior gives L1 regularization, and why does it produce sparse solutions?
:::details Solution
The negative log posterior is, up to constants, $\frac1{2\sigma^2}\sum_i(y_i-\boldsymbol{w}^\top\boldsymbol{x}_i)^2+\frac1{2\tau^2}\norm{\boldsymbol{w}}^2$. Multiply by $2\sigma^2$: the squared error plus $(\sigma^2/\tau^2)\norm{\boldsymbol{w}}^2$, so $\lambda=\sigma^2/\tau^2$: a tighter prior (smaller $\tau$) means more regularization. A Laplace prior $p(w)\propto e^{-|w|/b}$ gives the L1 penalty. The L1 ball has corners on the axes, and the penalty has a constant-magnitude gradient $\pm\lambda$ at every nonzero weight, so weights are pushed all the way to exactly zero (soft thresholding), whereas the L2 gradient $2\lambda w$ shrinks as $w\to0$.
:::
:::

:::exercise Question 5 (Weeks 3 and 4): coding, inverted dropout
Implement inverted dropout in NumPy (forward and backward), and explain why we scale at training time rather than at test time.
:::details Solution
```python
import numpy as np

class Dropout:
    def __init__(self, p=0.5, seed=0):
        self.p, self.rng, self.mask = p, np.random.default_rng(seed), None

    def forward(self, x, train=True):
        if not train or self.p == 0.0:
            return x                       # identity at test time
        keep = 1.0 - self.p
        self.mask = (self.rng.random(x.shape) < keep) / keep   # scale by 1/keep now
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask            # same mask: dropped units get zero gradient

x = np.ones((4, 5))
d = Dropout(0.5)
print(d.forward(x).mean())   # close to 1.0 in expectation
```
Each unit is kept with probability $1-p$, so $\E[x\cdot m]=x$ after scaling by $1/(1-p)$ and the expected activation matches between training and test. Scaling at training time leaves inference untouched, so there is no extra code or cost at test time.
:::
:::

### Section B: Core architectures (Weeks 5 to 8) {#exam-b}

:::exercise Question 6 (Week 5): shapes and parameters
A convolution takes a $3\times32\times32$ input with $k=5$, stride 2, padding 2, and 64 output channels. Give the output shape, the number of parameters, and the number of multiply-accumulates.
:::details Solution
Output size: $\lfloor(32+2\cdot2-5)/2\rfloor+1=\lfloor31/2\rfloor+1=16$, so the output is $64\times16\times16$. Parameters: $64\,(3\cdot5\cdot5+1)=64\cdot76=4864$. MACs: $16\cdot16\cdot64\cdot(3\cdot25)=1{,}228{,}800$ (about 1.2 million). The parameter count does not depend on the spatial size, which is the point of weight sharing.
:::
:::

:::exercise Question 7 (Week 6): why residual connections help
Compute $\partial\boldsymbol{x}_{l+1}/\partial\boldsymbol{x}_l$ for $\boldsymbol{x}_{l+1}=\boldsymbol{x}_l+F(\boldsymbol{x}_l)$, and extend it to $L$ blocks. Why does this mitigate vanishing gradients, and why does a plain stack of $L$ layers not?
:::details Solution
$\partial\boldsymbol{x}_{l+1}/\partial\boldsymbol{x}_l=\mathbf{I}+\partial F/\partial\boldsymbol{x}_l$. Across blocks, $\partial\boldsymbol{x}_L/\partial\boldsymbol{x}_l=\prod_{k=l}^{L-1}(\mathbf{I}+\mathbf{J}_k)$, whose expansion contains the term $\mathbf{I}$ (the identity path) plus terms with products of Jacobians. The loss gradient therefore always has a direct path $\partial\mathcal{L}/\partial\boldsymbol{x}_L$ to every earlier layer, without being multiplied by small Jacobians. In a plain stack the gradient is only the product $\prod_k\mathbf{J}_k$ of $L$ matrices; if their norms are below 1 the product decays exponentially in depth.
:::
:::

:::exercise Question 8 (Week 7): LSTM versus RNN gradient flow
Write the LSTM cell update and show which factor controls $\partial c_t/\partial c_{t-1}$. Why does this make long dependencies learnable, and when can it still fail?
:::details Solution
$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$, so (treating the gates as constants, the dominant direct path) $\partial c_t/\partial c_{t-1}=\diag(f_t)$. If the forget gate is near 1 the gradient is carried through almost unchanged, an additive path like a residual connection, rather than being multiplied by $\mathbf{W}_h^\top\diag(\tanh')$ at each step. It can still fail if the forget gates saturate near 0 (information is erased), if sequences are very long (even $0.99^{1000}\approx4\times10^{-5}$), or if gradients explode through other paths (so clip gradient norm). This is the motivation for attention, which provides direct paths between any two positions.
:::
:::

:::exercise Question 9 (Week 8): InfoNCE
Define the InfoNCE loss for a batch of $N$ positive pairs with cosine similarity and temperature $\tau$. What is the loss for a random model when $N=256$, and what do too small and too large $\tau$ do?
:::details Solution
For anchor $i$ with positive $i'$: $\mathcal{L}_i=-\log\dfrac{\exp(s_{ii'}/\tau)}{\sum_{j=1}^{N}\exp(s_{ij'}/\tau)}$, i.e. cross-entropy over the $N$ candidates where the positive is the correct class. For a random model all similarities are about equal, so the loss is $\log N=\log256\approx5.55$ (a useful sanity check). Small $\tau$ sharpens the softmax, concentrating the gradient on the hardest negatives, which can be unstable and push apart false negatives; large $\tau$ treats all negatives similarly and gives weak, less discriminative gradients.
:::
:::

### Section C: Transformers and language models (Weeks 9 and 10) {#exam-c}

:::exercise Question 10 (Week 9): why divide by the square root of d_k
Assume the entries of $\boldsymbol{q},\boldsymbol{k}\in\R^{d_k}$ are independent with mean 0 and variance 1. Compute the variance of $\boldsymbol{q}^\top\boldsymbol{k}$ and explain what it implies for the softmax.
:::details Solution
$\boldsymbol{q}^\top\boldsymbol{k}=\sum_{i=1}^{d_k}q_ik_i$ is a sum of $d_k$ independent terms, each with mean 0 and variance $\E[q_i^2]\E[k_i^2]=1$. So $\mathrm{Var}(\boldsymbol{q}^\top\boldsymbol{k})=d_k$ and its standard deviation is $\sqrt{d_k}$. Without scaling, scores grow with $d_k$, the softmax becomes nearly one-hot (saturated), its Jacobian $\diag(p)-pp^\top$ is nearly zero, and gradients vanish. Dividing by $\sqrt{d_k}$ restores unit variance.
:::
:::

:::exercise Question 11 (Week 9): parameter count of a Transformer
For GPT-2 small: $d=768$, $L=12$, vocabulary $V=50{,}257$, context 1024, FFN width $4d$, tied input and output embeddings. Estimate the parameter count.
:::details Solution
Per block: attention $4d^2$ (Q, K, V, output projections) plus FFN $2\cdot d\cdot4d=8d^2$, total $12d^2=12\cdot768^2\approx7.08$ M (biases and LayerNorm add under 0.1 M). Twelve blocks: about 85 M. Token embeddings: $50257\cdot768\approx38.6$ M (shared with the output layer). Positional embeddings: $1024\cdot768\approx0.8$ M. Total about 124 M, which matches the commonly quoted 124 M for GPT-2 small.
:::
:::

:::exercise Question 12 (Week 9): coding, causal self-attention
Write a PyTorch module for single-head causal self-attention for input of shape $(B,T,d)$. State the shape of every intermediate tensor.
:::details Solution
```python
import math
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d, d_k, max_len=1024):
        super().__init__()
        self.q = nn.Linear(d, d_k, bias=False)
        self.k = nn.Linear(d, d_k, bias=False)
        self.v = nn.Linear(d, d_k, bias=False)
        self.d_k = d_k
        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("mask", mask)

    def forward(self, x):                      # x: (B, T, d)
        B, T, _ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)          # each (B, T, d_k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)   # (B, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        w = scores.softmax(dim=-1)             # rows sum to 1; row t only sees positions <= t
        return w @ v                           # (B, T, d_k)

x = torch.randn(2, 5, 16)
print(CausalSelfAttention(16, 8)(x).shape)    # torch.Size([2, 5, 8])
```
The mask must be applied before the softmax (adding $-\infty$), not after, or rows would not sum to 1.
:::
:::

:::exercise Question 13 (Weeks 9 and 16): the KV cache
A decoder has $L=32$ layers and $d=4096$ (with full multi-head attention, so keys and values per token and layer have $d$ numbers each). For a 4096-token context in fp16, how much memory does the KV cache of one sequence take? How does multi-query or grouped-query attention change it?
:::details Solution
Per token and layer: $2d$ values (key and value) of 2 bytes, so $2\cdot4096\cdot2=16{,}384$ bytes. Times $L=32$ layers and $T=4096$ tokens: $16{,}384\cdot32\cdot4096=2{,}147{,}483{,}648$ bytes, which is 2 GiB for one sequence. With batch size 32 the cache alone is 64 GiB, which is why serving is memory-bound. Grouped-query attention shares each key/value head across a group of query heads; with $g$ KV heads instead of $h$ the cache shrinks by a factor $h/g$ (multi-query attention is $g=1$).
:::
:::

:::exercise Question 14 (Week 10): compute-optimal training
Use $C\approx6ND$ FLOPs. (a) How many FLOPs does it take to train a 7-billion-parameter model on 140 billion tokens? (b) At 40% utilization of an accelerator with peak $3\times10^{14}$ FLOP/s, how many accelerator-days is that? (c) What is the tokens-per-parameter ratio and how does it relate to the Chinchilla rule of thumb?
:::details Solution
(a) $C=6\cdot7\times10^9\cdot1.4\times10^{11}=5.88\times10^{21}$ FLOPs. (b) Effective throughput $0.4\cdot3\times10^{14}=1.2\times10^{14}$ FLOP/s, so $5.88\times10^{21}/1.2\times10^{14}=4.9\times10^{7}$ s $\approx567$ accelerator-days (for example about 9 days on 64 accelerators). (c) $D/N=20$ tokens per parameter, which is the approximate compute-optimal ratio reported by the Chinchilla study (Hoffmann et al., 2022). Practitioners who will serve a model widely often train smaller models on far more tokens than this, since inference cost is paid per query.
:::
:::

:::exercise Question 15 (Week 10): LoRA
For a weight $\mathbf{W}\in\R^{4096\times4096}$ and LoRA rank $r=8$: how many trainable parameters does LoRA add, what fraction of $\mathbf{W}$ is that, how is the update initialized, and why?
:::details Solution
LoRA learns $\Delta\mathbf{W}=\frac\alpha r\mathbf{B}\mathbf{A}$ with $\mathbf{B}\in\R^{4096\times8}$ and $\mathbf{A}\in\R^{8\times4096}$: $2\cdot4096\cdot8=65{,}536$ parameters, versus $16{,}777{,}216$ in $\mathbf{W}$, i.e. 0.39%. Initialization: $\mathbf{A}$ random (Gaussian), $\mathbf{B}=0$, so $\Delta\mathbf{W}=0$ at the start and the fine-tuned model begins exactly at the pretrained one, while gradients to $\mathbf{B}$ are nonzero (the gradient to $\mathbf{A}$ is zero at step 0 because $\mathbf{B}=0$, but it becomes nonzero after the first update). After training, $\mathbf{B}\mathbf{A}$ can be merged into $\mathbf{W}$ with no inference cost.
:::
:::

:::exercise Question 16 (Week 10): the role of the KL penalty in RLHF
The RLHF objective is $\max_\pi\E_{y\sim\pi}[r(x,y)]-\beta\,\KL(\pi\|\pi_{ref})$. What does the KL term prevent? What happens as $\beta\to0$ and as $\beta\to\infty$? How does DPO avoid training a separate reward model and running RL?
:::details Solution
The reward model is an imperfect proxy; unconstrained optimization exploits its errors (reward hacking) and drifts into text that scores high but is degenerate, and the KL term keeps the policy close to the reference (SFT) model where the reward model is accurate and the language stays fluent. As $\beta\to0$ the policy chases reward and over-optimizes; as $\beta\to\infty$ it stays at $\pi_{ref}$. The KL-regularized objective has the closed-form optimum $\pi^*(y|x)\propto\pi_{ref}(y|x)\exp(r(x,y)/\beta)$, so the reward can be written as $r=\beta\log(\pi^*/\pi_{ref})+\text{const}(x)$. Substituting this into the Bradley-Terry preference likelihood gives a loss on log-probability ratios of preferred and rejected responses, which is optimized with supervised gradient steps and no sampling.
:::
:::

:::exercise Question 17 (Week 10): decoding
Logits over 4 tokens are $[2.0,1.0,0.5,0.0]$. (a) Compute the probabilities at $T=1$. (b) What does $T=0.5$ do? (c) With top-$p$ of 0.8, which tokens remain?
:::details Solution
(a) $e^{2}=7.389$, $e^{1}=2.718$, $e^{0.5}=1.649$, $e^{0}=1$; sum $=12.756$; probabilities $[0.579,0.213,0.129,0.078]$. (b) $T=0.5$ doubles the logits to $[4,2,1,0]$: $e^4=54.60$, $e^2=7.389$, $e^1=2.718$, $1$, sum 65.71, probabilities $[0.831,0.112,0.041,0.015]$: the distribution is sharper (lower entropy, closer to greedy). (c) Sort descending and keep the smallest set whose cumulative probability is at least 0.8: $0.579$, then $0.579+0.213=0.792<0.8$, then $+0.129=0.921$, so the top three tokens remain and are renormalized; the last token is dropped.
:::
:::

### Section D: Vision and multimodal models (Week 11) {#exam-d}

:::exercise Question 18 (Week 11): ViT sequence length and cost
A ViT takes a $224\times224$ image with patch size $P=16$. (a) How many tokens enter the Transformer (with a class token)? (b) What happens to the number of tokens and to the attention cost if you move to $P=8$? (c) Why are the position embeddings a problem when you fine-tune at higher resolution?
:::details Solution
(a) $(224/16)^2=196$ patches plus 1 class token $=197$. (b) $P=8$ gives $28^2=784$ patches: $4\times$ the tokens. The attention matrix has $T^2$ entries, so attention cost grows $16\times$ (the linear layers grow $4\times$). (c) The learned position embeddings are tied to the original grid of $14\times14$ positions. At a higher resolution (more patches) the grid is larger, so the embeddings are usually 2-D interpolated, which works reasonably but is a source of train-test mismatch.
:::
:::

:::exercise Question 19 (Week 11): coding, the CLIP loss
Given normalized image embeddings `img` and text embeddings `txt`, both $(N,d)$, and a learnable log temperature, write the symmetric CLIP loss. What is the loss of a random model?
:::details Solution
```python
import torch
import torch.nn.functional as F

def clip_loss(img, txt, logit_scale):
    # img, txt: (N, d), L2-normalized; logit_scale = exp(log(1/tau)), a scalar tensor
    logits = logit_scale * img @ txt.t()          # (N, N): entry (i, j) = similarity(image i, text j)
    labels = torch.arange(img.size(0), device=img.device)   # matching pairs are on the diagonal
    loss_i = F.cross_entropy(logits, labels)       # image -> text
    loss_t = F.cross_entropy(logits.t(), labels)   # text -> image
    return (loss_i + loss_t) / 2

N, d = 8, 32
img = F.normalize(torch.randn(N, d), dim=-1)
txt = F.normalize(torch.randn(N, d), dim=-1)
print(clip_loss(img, txt, torch.tensor(1.0)).item())   # near log(8) = 2.08 for a random model
```
A random model gives about $\ln N$ (for $N=32{,}768$, about 10.4). Zero-shot classification then embeds the prompt "a photo of a {class}" for every class and picks the class with the highest cosine similarity to the image.
:::
:::

:::exercise Question 20 (Week 11): inductive bias and data
Explain why a ViT trained from scratch on 10 000 images tends to lose to a ResNet, but wins when pretrained on hundreds of millions of images. How do DeiT and MAE change this picture?
:::details Solution
Convolution hard-codes locality and translation equivariance; this prior is free information that makes learning efficient from small data. A ViT has only weak biases (patchification, and whatever is learned), so it must learn from data what a CNN assumes; with little data it overfits or underfits, with enough data the flexibility becomes an advantage and it scales better. DeiT recovers data efficiency on ImageNet-1k with heavy augmentation, regularization and distillation from a CNN teacher (the distillation token). MAE and other self-supervised methods supply a pretraining signal from unlabeled images, so the ViT can learn its own priors without labels. This is an empirical, not a proven, account.
:::
:::

### Section E: Generative models (Weeks 12 and 13) {#exam-e}

:::exercise Question 21 (Week 12): the ELBO
Derive the ELBO using Jensen's inequality, and show that the gap equals $\KL(q_\phi(z|x)\|p_\theta(z|x))$. For a diagonal Gaussian encoder $\mathcal{N}(\mu,\sigma^2)$ and prior $\mathcal{N}(0,1)$, give the closed form of the KL term and evaluate it for $\mu=1,\ \sigma=1$ in one dimension.
:::details Solution
$\log p_\theta(x)=\log\int p_\theta(x,z)\,dz=\log\E_{q}\Big[\frac{p_\theta(x,z)}{q_\phi(z|x)}\Big]\ge\E_q\Big[\log\frac{p_\theta(x,z)}{q_\phi(z|x)}\Big]=\E_q[\log p_\theta(x|z)]-\KL(q_\phi(z|x)\|p(z))$ by Jensen (log is concave). Exactly, $\log p_\theta(x)=\mathrm{ELBO}+\KL(q_\phi(z|x)\|p_\theta(z|x))$, so the gap is that KL and is zero when $q$ equals the true posterior. For diagonal Gaussians $\KL=\frac12\sum_j(\mu_j^2+\sigma_j^2-1-\ln\sigma_j^2)$. For $\mu=1,\sigma=1$: $\frac12(1+1-1-0)=0.5$.
:::
:::

:::exercise Question 22 (Week 12): the reparameterization trick
Why can't we backpropagate through $z\sim\mathcal{N}(\mu_\phi,\sigma_\phi^2)$ directly, and how does $z=\mu_\phi+\sigma_\phi\epsilon$ fix it? Is it possible for discrete latents?
:::details Solution
Sampling is a stochastic node, and the gradient of an expectation with respect to the distribution's parameters, $\nabla_\phi\E_{z\sim q_\phi}[f(z)]$, is not an expectation of $\nabla f$. Writing $z=\mu_\phi+\sigma_\phi\epsilon$ with $\epsilon\sim\mathcal{N}(0,1)$ moves the randomness to a parameter-free input, so $\nabla_\phi\E_\epsilon[f(\mu_\phi+\sigma_\phi\epsilon)]=\E_\epsilon[\nabla_\phi f(\cdot)]$ and ordinary backprop works, with lower variance than score-function estimators. For discrete variables there is no differentiable reparameterization; one uses score-function (REINFORCE) estimators with baselines, or relaxations like Gumbel-softmax (biased), or straight-through estimators (as in VQ-VAE).
:::
:::

:::exercise Question 23 (Week 12): the optimal discriminator
For fixed $G$ with generated density $p_g$, derive the optimal discriminator of the GAN objective and the value of the objective at the optimum in terms of the Jensen-Shannon divergence.
:::details Solution
$V(D,G)=\int\big[p_{data}(x)\log D(x)+p_g(x)\log(1-D(x))\big]dx$. Pointwise, $f(D)=a\log D+b\log(1-D)$ with $a=p_{data}(x)$, $b=p_g(x)$ is maximized where $a/D-b/(1-D)=0$, i.e. $D^*(x)=\dfrac{p_{data}(x)}{p_{data}(x)+p_g(x)}$. Substituting, $V(D^*,G)=\E_{p_{data}}\log\frac{p_{data}}{p_{data}+p_g}+\E_{p_g}\log\frac{p_g}{p_{data}+p_g}=-\log4+2\,\mathrm{JSD}(p_{data}\|p_g)$ (using $\KL(p\|\tfrac{p+q}2)$ terms). So the generator minimizes the JSD, with global optimum $p_g=p_{data}$ giving $D^*=1/2$ and $V=-\log4$. In practice the JSD gives vanishing gradients when the supports are disjoint, which motivates the Wasserstein distance.
:::
:::

:::exercise Question 24 (Week 13): coding, the diffusion forward process
Derive $q(x_t|x_0)$ in closed form for $x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\epsilon_t$, then write `q_sample` and the training loss for a linear schedule.
:::details Solution
Let $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s\le t}\alpha_s$. Then $x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t$. Substituting $x_{t-1}$ gives $x_t=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}+\sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-1}+\sqrt{1-\alpha_t}\epsilon_t$. The two noise terms are independent Gaussians with total variance $\alpha_t(1-\alpha_{t-1})+1-\alpha_t=1-\alpha_t\alpha_{t-1}$, so they merge into a single noise term. By induction $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$, i.e. $q(x_t|x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)\mathbf{I})$.
```python
import torch
import torch.nn.functional as F

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas_bar = torch.cumprod(1.0 - betas, dim=0)           # (T,)

def q_sample(x0, t, noise):
    # x0: (B, C, H, W), t: (B,) long, noise: same shape as x0
    ab = alphas_bar[t].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

def diffusion_loss(model, x0):
    t = torch.randint(0, T, (x0.size(0),), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise)
    return F.mse_loss(model(x_t, t), noise)               # predict epsilon

x0 = torch.randn(4, 1, 8, 8)
t = torch.tensor([0, 10, 500, 999])
print(q_sample(x0, t, torch.randn_like(x0)).std(dim=(1, 2, 3)))
```
The printed standard deviations increase toward 1 as $t$ grows: by $t=T$ the signal is essentially destroyed.
:::
:::

:::exercise Question 25 (Week 13): classifier-free guidance
State the guided noise prediction. What does increasing the guidance weight do to sample fidelity, diversity and sampling cost? Why does it work as a "sharpening" of the conditional distribution?
:::details Solution
$\tilde\epsilon=\epsilon_\theta(x_t,\varnothing)+(1+w)\big(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\big)$, equivalently $(1+w)\epsilon_\theta(x_t,c)-w\,\epsilon_\theta(x_t,\varnothing)$. The model is trained with the condition randomly dropped (for example 10% of the time) so a single network provides both terms. Since $\epsilon$ is proportional to the negative score, the extrapolation corresponds to the score of $p(x|c)\,\big[p(c|x)\big]^{w}$ up to normalization: it moves samples toward regions where the condition is more identifiable. Larger $w$ increases fidelity and prompt adherence and reduces diversity (and can saturate colors and create artifacts); it doubles the network evaluations per step (conditional and unconditional passes, usually batched).
:::
:::

:::exercise Question 26 (Weeks 12 and 13): choosing a generative family
A product team needs (i) image generation within 50 ms, (ii) exact bits-per-dimension for compression, (iii) a smooth latent for interpolation and editing. Which family would you pick for each, and what is the main drawback?
:::details Solution
(i) A GAN (a single forward pass) or a distilled few-step diffusion model; the drawback is training instability and mode dropping for GANs, quality loss for aggressive distillation. (ii) An autoregressive model or normalizing flow gives exact $\log p(x)$; the drawback is slow sequential sampling (autoregressive) or architectural constraints (flows). (iii) A VAE (or the autoencoder of a latent diffusion model) has an encoder and a smooth latent space; the drawback is blurrier samples from the pixel-level likelihood, and posterior collapse risk. A diffusion model has no compact latent by itself and needs many steps to sample.
:::
:::

### Section F: Graph neural networks (Week 14) {#exam-f}

:::exercise Question 27 (Week 14): permutation equivariance
Show that a GCN layer $\mathbf{H}'=\sigma(\hat{\mathbf{A}}\mathbf{H}\mathbf{W})$ with $\hat{\mathbf{A}}=\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$ is permutation equivariant, and explain why a graph-level readout must be invariant.
:::details Solution
Relabel the nodes with a permutation matrix $\mathbf{P}$: $\mathbf{A}\to\mathbf{P}\mathbf{A}\mathbf{P}^\top$, $\mathbf{H}\to\mathbf{P}\mathbf{H}$. Then $\tilde{\mathbf{A}}\to\mathbf{P}\tilde{\mathbf{A}}\mathbf{P}^\top$ (since $\mathbf{P}\mathbf{I}\mathbf{P}^\top=\mathbf{I}$), the degree matrix becomes $\mathbf{P}\tilde{\mathbf{D}}\mathbf{P}^\top$, and so $\hat{\mathbf{A}}\to\mathbf{P}\hat{\mathbf{A}}\mathbf{P}^\top$. The layer outputs $\sigma(\mathbf{P}\hat{\mathbf{A}}\mathbf{P}^\top\mathbf{P}\mathbf{H}\mathbf{W})=\mathbf{P}\,\sigma(\hat{\mathbf{A}}\mathbf{H}\mathbf{W})$ because $\mathbf{P}^\top\mathbf{P}=\mathbf{I}$ and $\sigma$ is elementwise: output rows are permuted the same way as the input. For a graph-level prediction the answer must not depend on the arbitrary node ordering, so the readout must be invariant (sum, mean, max or an attention-based pooling over nodes).
:::
:::

:::exercise Question 28 (Week 14): aggregators and expressiveness
Give an example in which mean and max aggregation cannot distinguish two neighborhoods but sum can. What does this have to do with the Week 14 expressiveness result?
:::details Solution
Neighborhood 1 has two neighbors with identical features $a$: multiset $\{a,a\}$; neighborhood 2 has one neighbor with feature $a$: $\{a\}$. Mean gives $a$ in both cases, max gives $a$ in both, but sum gives $2a$ versus $a$. The GIN analysis (Xu et al., 2019) shows that an injective multiset function is needed to match the discriminative power of the 1-WL isomorphism test, and sum (followed by an MLP) can be injective on countable multisets whereas mean and max cannot. Hence message-passing GNNs with mean or max are strictly weaker, and no standard message-passing GNN is more powerful than 1-WL.
:::
:::

:::exercise Question 29 (Week 14): over-smoothing
Why do node representations become indistinguishable as you stack many GCN layers? Give two remedies and one tradeoff.
:::details Solution
A GCN layer applies a low-pass filter $\hat{\mathbf{A}}$; repeated application $\hat{\mathbf{A}}^k\mathbf{H}$ converges toward the dominant eigenvector of $\hat{\mathbf{A}}$ (constant on each connected component, scaled by degree), so node features lose discriminative information. Remedies: residual or skip connections (as in JKNet or GCNII), fewer layers, normalization layers such as PairNorm, DropEdge, and attention or jumping-knowledge readouts. The tradeoff: depth is what allows information to travel $k$ hops, so cutting layers worsens long-range reasoning (over-squashing is the related, separate bottleneck); alternatives such as graph Transformers or rewiring address range directly.
:::
:::

### Section G: Reinforcement learning (Week 15) {#exam-g}

:::exercise Question 30 (Week 15): TD targets by hand
A transition has $r=1$, $\gamma=0.9$, $Q(s,a)=4$. At $s'$ the online network gives $Q=[3,1]$ and the target network gives $Q=[2,5]$ for the two actions. Compute the DQN target, the TD error and the Q-learning update with $\alpha=0.1$ (tabular style), and the Double DQN target.
:::details Solution
DQN target uses the target network's max: $y=1+0.9\cdot5=5.5$. TD error $\delta=5.5-4=1.5$, update $Q\leftarrow4+0.1\cdot1.5=4.15$. Double DQN chooses the action with the online network ($\arg\max=$ action 0) and evaluates it with the target network: $y=1+0.9\cdot2=2.8$, so $\delta=-1.2$. Decoupling selection from evaluation removes the positive bias that the max over noisy estimates introduces.
:::
:::

:::exercise Question 31 (Week 15): baselines in policy gradient
Show that subtracting a state-dependent baseline $b(s)$ from the return does not bias the policy-gradient estimator, and say what the best baseline aims to do.
:::details Solution
The extra term is $\E_{a\sim\pi_\theta}[\nabla_\theta\log\pi_\theta(a|s)\,b(s)]=b(s)\sum_a\pi_\theta(a|s)\nabla_\theta\log\pi_\theta(a|s)=b(s)\sum_a\nabla_\theta\pi_\theta(a|s)=b(s)\nabla_\theta\sum_a\pi_\theta(a|s)=b(s)\nabla_\theta1=0$. So the estimator stays unbiased (the baseline cannot depend on the action taken, or the argument fails). A good baseline reduces variance; the value function $V^\pi(s)$ is the standard choice, and $G_t-V(s_t)$ (or the advantage estimated by GAE) is the learning signal for actor-critic methods.
:::
:::

:::exercise Question 32 (Week 15): the PPO clipped objective
With $\epsilon=0.2$ and probability ratio $r=1.5$, evaluate $\min(r\hat A,\ \mathrm{clip}(r,0.8,1.2)\hat A)$ for $\hat A=+2$ and for $\hat A=-2$, and say in which case the gradient is zero.
:::details Solution
For $\hat A=+2$: $r\hat A=3$ and $\mathrm{clip}(r)\hat A=2.4$; the minimum is $2.4$, the clipped term, which is constant in $\theta$, so the gradient is zero: once the action's probability has already risen by more than 20%, there is no more incentive to increase it. For $\hat A=-2$: $r\hat A=-3$, clipped term $=-2.4$; the minimum is $-3$, the unclipped term, so the gradient flows and pushes the ratio back down. The objective is pessimistic: it removes the incentive for overly large policy changes in the profitable direction but not in the harmful one.
:::
:::

:::exercise Question 33 (Week 15): find the bugs
The following DQN update has at least three problems. Find them.
```python
q = qnet(s).gather(1, a)                    # s: (B, 4), a: (B, 1)
q_next = qnet(s2).max(dim=1).values          # (B,)
target = r + gamma * q_next                  # r: (B,)
loss = ((q - target) ** 2).mean()
loss.backward()
opt.step()
```
:::details Solution
(1) The target is computed with the online network and with gradients, so the target moves with every update and gradients flow through it. Use a separate, slowly updated target network and `torch.no_grad()` (or `.detach()`). (2) Terminal transitions are not masked: the target must be `r + gamma * (1 - done) * q_next`. (3) Shape bug: `q` is $(B,1)$ and `target` is $(B,)$, so `q - target` broadcasts to $(B,B)$ and silently averages the wrong quantity; use `q.squeeze(1)` or `gather(...).squeeze(1)`. Also missing: `opt.zero_grad()` before `backward()`, and a Huber loss and gradient clipping would improve stability.
:::
:::

### Section H: Efficiency and scale (Week 16) {#exam-h}

:::exercise Question 34 (Week 16): roofline reasoning
An accelerator has 300 TFLOP/s of fp16 compute and 2 TB/s of memory bandwidth. (a) What is the ridge-point arithmetic intensity? (b) For an fp16 matrix product $(M\times K)\cdot(K\times N)$ with $M=N=K=4096$, is it compute or memory bound? (c) What about the matrix-vector case $M=1$, $K=N=4096$, as in single-sequence decoding?
:::details Solution
(a) Ridge point $=3\times10^{14}/2\times10^{12}=150$ FLOP/byte. (b) FLOPs $=2MKN$; bytes $=2(MK+KN+MN)$ (2 bytes per fp16 element). For $M=N=K=n$: intensity $=2n^3/(6n^2)=n/3=1365$ FLOP/byte, well above 150, so compute bound. (c) For $M=1$: FLOPs $=2KN\approx3.4\times10^7$; bytes $\approx2(K+KN+N)\approx3.4\times10^7$; intensity $\approx1$ FLOP/byte, far below 150, so memory bound: the whole weight matrix is read to do one multiply-add per weight. Attainable throughput is about $1\times2$ TFLOP/s, under 1% of peak, which is why decoding is bandwidth-limited and why batching, quantization and speculative decoding help.
:::
:::

:::exercise Question 35 (Week 16): memory for training
How much memory do the parameters, gradients and Adam states take for a 7B-parameter model trained with mixed precision (bf16 compute, fp32 master weights and moments)? What does ZeRO stage 3 across 8 GPUs do to it, and what is not included?
:::details Solution
Per parameter: 2 bytes (bf16 weight) $+$ 2 bytes (bf16 gradient) $+$ 4 (fp32 master weight) $+$ 4 ($m$) $+$ 4 ($v$) $=16$ bytes. For $7\times10^9$ parameters: $112\times10^9$ bytes, about 112 GB (104 GiB). ZeRO-3 / FSDP shards all three (optimizer states, gradients, parameters) across the 8 GPUs: about 14 GB per GPU for these states. Not included: activations (which depend on batch, sequence length and checkpointing), temporary buffers and the communication needed to all-gather each layer's parameters just in time.
:::
:::

:::exercise Question 36 (Week 16): gradient accumulation
(a) Write the accumulation loop to emulate a batch of 256 with micro-batches of 32. (b) For which layer does this differ from a true batch of 256?
:::details Solution
(a)
```python
accum = 8                                   # 256 / 32
opt.zero_grad(set_to_none=True)
for i, (x, y) in enumerate(loader):         # loader yields micro-batches of 32
    loss = criterion(model(x), y) / accum   # divide so the summed gradient is the mean
    loss.backward()                         # gradients add up in .grad
    if (i + 1) % accum == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
```
(b) For layers whose forward pass couples the examples in the batch: BatchNorm computes statistics from each micro-batch of 32, not 256, and is therefore not equivalent (use GroupNorm or LayerNorm, or sync/accumulate statistics). Dropout masks and in-batch contrastive losses (CLIP, InfoNCE with in-batch negatives) also differ: the number of negatives is that of the micro-batch.
:::
:::

### Section I: Trustworthy deep learning (Week 17) {#exam-i}

:::exercise Question 37 (Week 17): FGSM from first principles
Derive FGSM from a first-order Taylor expansion of the loss for an $\ell_\infty$ budget $\epsilon$, and state the corresponding step for an $\ell_2$ budget.
:::details Solution
$\mathcal{L}(\boldsymbol{x}+\boldsymbol{\delta})\approx\mathcal{L}(\boldsymbol{x})+\boldsymbol{g}^\top\boldsymbol{\delta}$ with $\boldsymbol{g}=\nabla_{\boldsymbol{x}}\mathcal{L}$. Maximize over $\norm{\boldsymbol{\delta}}_\infty\le\epsilon$: each coordinate is independent and $g_i\delta_i$ is maximized by $\delta_i=\epsilon\,\mathrm{sign}(g_i)$, giving the increase $\epsilon\norm{\boldsymbol{g}}_1$. For an $\ell_2$ ball the maximizer is $\boldsymbol{\delta}=\epsilon\,\boldsymbol{g}/\norm{\boldsymbol{g}}_2$ (Cauchy-Schwarz) with increase $\epsilon\norm{\boldsymbol{g}}_2$. The linearity explanation notes that in $d$ dimensions the $\ell_1$ norm of a gradient can be large even for a tiny per-coordinate budget, so many small coordinated changes add up.
:::
:::

:::exercise Question 38 (Week 17): expected calibration error
A classifier makes 10 predictions in two confidence bins. Bin A has 4 predictions with mean confidence 0.9 and 3 correct. Bin B has 6 predictions with mean confidence 0.6 and 3 correct. Compute the ECE and say whether the model is over- or underconfident.
:::details Solution
Bin A: accuracy $0.75$, confidence $0.90$, gap $0.15$, weight $0.4$. Bin B: accuracy $0.50$, confidence $0.60$, gap $0.10$, weight $0.6$. $\mathrm{ECE}=0.4\cdot0.15+0.6\cdot0.10=0.06+0.06=0.12$. In both bins confidence exceeds accuracy: overconfident. Temperature scaling with $T>1$ would soften the probabilities and reduce the gap without changing accuracy.
:::
:::

:::exercise Question 39 (Week 17): split conformal prediction
You have $n=19$ calibration nonconformity scores and want 90% coverage. (a) Which order statistic is the threshold? (b) State the guarantee and its assumption. (c) Why is it only a marginal guarantee?
:::details Solution
(a) $k=\lceil(n+1)(1-\alpha)\rceil=\lceil20\cdot0.9\rceil=18$: use the 18th smallest score as the threshold $\hat q$ and include in the prediction set every label whose score is at most $\hat q$. (b) If the calibration and test points are exchangeable, $P(Y_{n+1}\in C(X_{n+1}))\ge1-\alpha$; this holds for any model and any finite $n$ (the proof: the test score is equally likely to be at any rank among the $n+1$ scores). (c) The probability is over the draw of calibration set and test point jointly, so coverage can still be much lower for particular inputs, classes or subgroups; conditional coverage is not guaranteed. Distribution shift also breaks exchangeability.
:::
:::

:::exercise Question 40 (Week 17): DP-SGD
Describe one DP-SGD step. Why must per-example gradients be clipped, and what is the role of the noise multiplier? What does the privacy budget depend on?
:::details Solution
For a sampled batch: compute each example's gradient $g_i$, clip it to norm at most $C$ ($\bar g_i=g_i/\max(1,\norm{g_i}/C)$), sum, add Gaussian noise $\mathcal{N}(0,\sigma^2C^2\mathbf{I})$, divide by the batch size, and take an optimizer step. Clipping bounds the influence (sensitivity) of any single example on the sum by $C$; without a bound no finite noise level could hide an individual. The noise scale is proportional to $\sigma C$; larger $\sigma$ means more privacy and lower accuracy. The $(\varepsilon,\delta)$ guarantee follows from composition over steps with amplification by subsampling, computed by a moments or Renyi accountant, and depends on $\sigma$, the sampling rate $q=B/N$, the number of steps, and $\delta$. Training longer or with a smaller $\sigma$ spends more budget.
:::
:::

:::exercise Question 41 (Week 17): fairness criteria
Define demographic parity and equalized odds. Explain why, when base rates differ between groups, a non-perfect classifier cannot satisfy calibration within groups and equal false positive and false negative rates at once.
:::details Solution
Demographic parity: $P(\hat Y=1|A=a)$ is equal across groups $a$. Equalized odds: $P(\hat Y=1|Y=y,A=a)$ is equal across groups for $y\in\{0,1\}$ (equal TPR and FPR). Calibration within groups (predictive parity): $P(Y=1|S=s,A=a)$ is the same function of the score for every group. Chouldechova (2017) and Kleinberg, Mullainathan and Raghavan (2016) show that with unequal base rates $p_a\ne p_b$ and imperfect prediction these conditions are mutually incompatible: for example, the identity $\mathrm{FPR}=\frac{p}{1-p}\frac{1-\mathrm{PPV}}{\mathrm{PPV}}\mathrm{TPR}$ ties the error rates to the base rate $p$, so equal PPV and equal TPR force unequal FPR. The choice of criterion is therefore a policy decision, not a purely technical one.
:::
:::

:::exercise Question 42 (Week 17): coding, temperature scaling
Fit a temperature on validation logits by minimizing NLL, and explain why the test accuracy cannot change.
:::details Solution
```python
import torch
import torch.nn.functional as F

def fit_temperature(logits, labels):
    # logits: (N, K) from a frozen model on a held-out validation set
    log_T = torch.zeros(1, requires_grad=True)          # parametrize T = exp(log_T) > 0
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_T.exp(), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return log_T.exp().item()

torch.manual_seed(0)
logits = torch.randn(1000, 5) * 3                        # overconfident toy logits
labels = torch.randint(0, 5, (1000,))
print(fit_temperature(logits, labels))                   # T > 1: random labels, so it softens a lot
```
Dividing all logits by $T>0$ preserves their order, hence the argmax and the accuracy. Only the confidence changes. Fit on validation data, never on the test set.
:::
:::

### Section J: Reporting and the scientific method (Week 18) {#exam-j}

:::exercise Question 43 (Week 18): is the claim supported?
A team reports test accuracy of method A as 71.2, 70.8, 71.5 and of method B as 71.9, 71.0, 72.1 (three seeds each) and concludes "B is better". Evaluate the claim.
:::details Solution
Means: $\bar x_A=71.17$, $\bar x_B=71.67$, a gap of $0.50$. Sample standard deviations: $s_A\approx0.35$, $s_B\approx0.59$. The standard error of the difference is $\sqrt{0.35^2/3+0.59^2/3}\approx0.39$, so Welch's $t\approx0.50/0.39\approx1.3$ with about 3.3 degrees of freedom; the 5% critical value is around 3. The difference is not significant, and each method's own 95% interval ($\pm4.30\,s/\sqrt3$, about $\pm0.87$ for A and $\pm1.46$ for B) comfortably contains the other's mean. The right statement is "B is higher in these three runs, but three seeds cannot distinguish a 0.5-point gap from noise." The fix is more seeds (the explorer above shows how quickly the interval narrows), a paired analysis if the seeds are matched, and a check that both methods got the same tuning effort.
:::
:::

## Exercises {#exercises}

The practice exam above is the main exercise set for this week. These additional exercises focus on the project and on the cross-cutting skills.

:::exercise Exercise 1: audit your own project (conceptual)
Fill in the sentence "We show that ___, by measuring ___, against ___." for your project. Then list the three weakest points of your evidence and what experiment would address each.
:::details Solution
There is no single answer; a good response is specific. Example: "We show that cosine decay with warmup improves a small ViT on CIFAR-10 by about 1 point (claim), by measuring top-1 test accuracy over 5 seeds with a 95% interval (metric), against the same ViT with a constant learning rate and a ResNet-18 baseline (baselines)." Typical weak points and fixes: only 3 seeds (run 5 to 10); the baseline learning rate was not tuned (give it the same search budget); the hyperparameters were chosen by looking at test accuracy (re-select on a validation split and evaluate once on test); no failure analysis (inspect and slice the 50 worst examples).
:::
:::

:::exercise Exercise 2: sample size for a target interval (derivation)
Your per-seed standard deviation is about $\sigma=0.6$ points. Using the normal approximation (multiplier 1.96), how many seeds do you need for a 95% confidence interval of half-width at most $0.3$ points? Is that the number you need to *detect* a true gap of 0.3 points between two methods?
:::details Solution
Half-width $=1.96\,\sigma/\sqrt n\le0.3\Rightarrow\sqrt n\ge1.96\cdot0.6/0.3=3.92\Rightarrow n\ge15.4$, so $n=16$ (using the $t$ multiplier for such $n$ gives about 17). For detecting a difference between two independent methods, the standard error of the gap is $\sigma\sqrt{2/n}$, so to have the interval half-width of the gap below the gap itself ($0.3$) you need $1.96\cdot0.6\sqrt{2/n}\le0.3$, i.e. $n\ge2\cdot15.4\approx31$ seeds per method. A reliable detection (power 80%) needs roughly a factor of 2 more than that threshold-type requirement. This is why small effects are expensive to establish and why paired designs or larger effects are preferred.
:::
:::

:::exercise Exercise 3: coding, an ablation grid with seeds
Extend the `Run` harness so that a function `sweep(train_fn, configs, seeds)` runs every configuration for every seed and prints a table of mean, standard deviation and CI per configuration. Which seeds should you share across configurations?
:::details Solution
```python
import itertools
from dataclasses import replace
import numpy as np
from tracking import Config, Run
from seed_stats import summarize

def sweep(train_fn, configs, seeds):
    """train_fn(cfg, run) -> final test metric. Returns {config name: [metric per seed]}."""
    results = {}
    for cfg, seed in itertools.product(configs, seeds):
        c = replace(cfg, seed=seed)
        run = Run(c)
        metric = train_fn(c, run)
        run.finish(test_metric=metric)
        results.setdefault(cfg.name, []).append(metric)
    print(f"{'config':<20}{'mean':>8}{'std':>8}{'ci95':>8}")
    for name, xs in results.items():
        s = summarize(xs)
        print(f"{name:<20}{s['mean']:>8.2f}{s['std']:>8.2f}{s['ci_half']:>8.2f}")
    return results

# Example with a fake training function whose metric depends on lr and the seed
def fake_train(cfg, run):
    rng = np.random.default_rng(cfg.seed)
    return 90 + (0.5 if cfg.lr == 3e-4 else 0.0) + 0.4 * rng.normal()

cfgs = [Config(name="lr1e-3", lr=1e-3), Config(name="lr3e-4", lr=3e-4)]
sweep(fake_train, cfgs, seeds=range(5))
```
Use the same list of seeds for every configuration, so comparisons can be paired and differences in data order or initialization do not masquerade as differences between configurations.
:::
:::

:::exercise Exercise 4: reproduce a figure (challenging)
Choose one published figure or table from a Week 9 to 17 paper that your hardware can reproduce at small scale (for example, the effect of $1/\sqrt{d_k}$ scaling on attention entropy, the Wasserstein-versus-JS gradient behavior on disjoint supports, or temperature-scaling reliability diagrams). Plan the reproduction: what is the claim, what can you scale down without changing its meaning, what would count as success, and what would you report if you failed?
:::details Solution
A good plan states the claim in one sentence and the *invariants* you must keep (for the attention example: unit-variance queries and keys, a range of $d_k$, softmax over the same number of positions) and the things you can shrink (the sequence length, the number of heads, the training length). Success criterion: a quantitative one, such as "the mean attention entropy stays near $\log T$ with the scaling and falls toward 0 without it as $d_k$ grows from 16 to 1024, over 10 random draws". If the figure does not reproduce, report it as a finding with the details: which settings differ from the paper, which you could not determine from the text, and what you tried. A failed reproduction with a careful log is a legitimate and useful result; an unexamined mismatch is not.
:::
:::

:::exercise Exercise 5: design the evaluation (what would happen if...)
You fine-tune a pretrained model on a 5,000-example dataset and report test accuracy from the best validation checkpoint chosen over 30 hyperparameter configurations, each with one seed. A reviewer suspects the reported gain over the baseline is inflated. List three mechanisms that could inflate it and one protocol that removes them.
:::details Solution
(1) Selection bias: the best of 30 noisy single-seed runs is on average higher than the true performance of the recipe (winner's curse), so the reported number is optimistic even on a validation set, and more so if selection is done on test data. (2) The baseline received less tuning, so part of the gap is tuning effort. (3) Test set noise: with a small test set the standard error of accuracy is large, and any small difference can be noise; leakage between pretraining data and the test set is also possible. Protocol: split a separate validation set; choose hyperparameters on it with a fixed budget for every method; retrain the chosen configuration with 5 or more seeds; evaluate once on the test set; report mean and interval for both methods; optionally use nested cross-validation if data are scarce.
:::
:::

## Common mistakes and debugging tips {#mistakes}

Mistakes that cost points on the final project and in the exam:

- **Single-seed claims.** Symptom: "B beats A by 0.3". Fix: at least 3 to 5 seeds, intervals, and cautious wording.
- **Baseline neglect.** Symptom: huge gains over a default-hyperparameter baseline. Fix: tune the baseline with the same budget; include a simple non-deep baseline.
- **Test-set contamination.** Symptoms: validation and test curves that are suspiciously close, or results that vanish on fresh data. Fix: freeze the test set, look at it once, check for duplicates across splits.
- **Cherry-picked figures.** Qualitative examples that are best-case. Fix: random, uncurated examples and failure cases.
- **An over-long talk.** Symptom: the last three slides are rushed. Fix: rehearse with a timer and cut content.
- **Unexplained numbers.** A table with no stated metric, split or seed count. Fix: captions that say what, on which data, over how many runs.
- **A README that does not run.** Symptom: missing data download step, unpinned versions. Fix: a clean-clone test on another machine or a fresh Colab session.
- **Exam: skipping units and shapes.** Write the shape of every tensor and the units of every quantity; most calculation errors show up there.
- **Exam: derivations without definitions.** State what every symbol means; partial credit depends on it.
- **Exam: memorizing without understanding.** Many questions change one assumption (for example, "what if the layer used a sigmoid instead of ReLU?"); practice asking that question of every result.

## Key takeaways {#takeaways}

:::key Key takeaways
- A project talk makes one claim, spends most of its time on evidence, and shows uncertainty and failures. Twelve minutes means eight to ten slides.
- A fair comparison needs a strong baseline with equal tuning effort, ablations that isolate each component, and several seeds. Intervals shrink only like $1/\sqrt n$; a single seed can pick the wrong winner a third of the time for a 0.3-point effect.
- Report mean and interval over seeds with the multiplier $t_{n-1,0.975}$, say what the error bar is, and consider paired analyses and test-set sampling error separately.
- Reproducibility means recording code, config, data, environment and randomness together for every run, and shipping a README that regenerates every number in the report.
- The course in one line: a loss (usually a log-likelihood or bound), a parameterized function with the right inductive bias, an optimizer, data, and the scientific discipline to know whether it worked.
- Debug from simple to complex: overfit one batch, check the initial loss, inspect data, then tune.
- To study for the exam, retrieve rather than reread: derive, compute and code from a blank page, and fix what you miss.
:::

**Part IV: next steps**

## How to read research papers {#reading-papers}

By now you have read and presented papers (see [How to read and present a paper](week-10.html#reading-papers)). The skill that separates efficient readers from exhausted ones is *multi-pass reading*, which follows Keshav's "How to Read a Paper" (2007).

{{figure: week18-paper-passes | Three passes over a paper. Most papers deserve only the first pass; a few deserve the second; very few deserve the third.}}

During pass 2 keep a short written summary in a fixed template; it forces understanding and makes the paper searchable later:

1. **Problem.** What is being solved, and why is it hard?
2. **Key idea.** One or two sentences, in your own words.
3. **Method.** The model, loss and training procedure, with notation.
4. **Evidence.** Which experiments support which claims? What are the baselines, the number of seeds, and the effect size relative to noise?
5. **Limitations.** Stated by the authors and found by you.
6. **Connections.** Which course weeks and other papers does it relate to?

Questions to ask in the critique pass: Is the baseline strong and tuned? Is the compute budget matched? Does the ablation isolate the claimed cause? Is the improvement larger than seed noise? Does the benchmark measure what the title claims? Could the test set have been seen in pretraining? Are there negative results hidden in the appendix? What would falsify the central claim?

Practical advice: read the abstract, the figures and the conclusion first; read the appendix for hyperparameters and extra results; follow the references of a paper you like *backwards* (what it builds on) and the citations *forwards* (what built on it, via Google Scholar or Semantic Scholar). Keep a reading log. Join or start a reading group: explaining a paper aloud exposes gaps better than rereading. Skim arXiv regularly (the daily lists in cs.LG, cs.CL, cs.CV), but prefer depth over breadth; a handful of papers read well beats fifty skimmed.

## A reading list by topic {#reading-list}

The following papers are landmarks, chosen because they are readable and because they shaped the field. All are real and verifiable; years are the publication or first arXiv year.

**Foundations and optimization**

- Rumelhart, Hinton and Williams, "Learning representations by back-propagating errors", *Nature*, 1986.
- Kingma and Ba, "Adam: A method for stochastic optimization", ICLR 2015 (arXiv:1412.6980).
- Ioffe and Szegedy, "Batch normalization: accelerating deep network training by reducing internal covariate shift", ICML 2015.
- Srivastava et al., "Dropout: a simple way to prevent neural networks from overfitting", JMLR 2014.
- Zhang, Bengio, Hardt, Recht and Vinyals, "Understanding deep learning requires rethinking generalization", ICLR 2017.

**Vision**

- Krizhevsky, Sutskever and Hinton, "ImageNet classification with deep convolutional neural networks", NeurIPS 2012.
- He, Zhang, Ren and Sun, "Deep residual learning for image recognition", CVPR 2016.
- Dosovitskiy et al., "An image is worth 16x16 words: transformers for image recognition at scale", ICLR 2021.
- He et al., "Masked autoencoders are scalable vision learners", CVPR 2022.
- Radford et al., "Learning transferable visual models from natural language supervision" (CLIP), ICML 2021.

**Sequence models, Transformers and language models** (the essential trio is marked)

- **Vaswani et al., "Attention is all you need", NeurIPS 2017.**
- Devlin, Chang, Lee and Toutanova, "BERT: pre-training of deep bidirectional transformers for language understanding", NAACL 2019.
- Brown et al., "Language models are few-shot learners" (GPT-3), NeurIPS 2020.
- Kaplan et al., "Scaling laws for neural language models", 2020 (arXiv:2001.08361).
- Hoffmann et al., "Training compute-optimal large language models" (Chinchilla), 2022 (arXiv:2203.15556).
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), NeurIPS 2022.
- Rafailov et al., "Direct preference optimization: your language model is secretly a reward model", NeurIPS 2023.
- Hu et al., "LoRA: low-rank adaptation of large language models", ICLR 2022.

**Generative models**

- Kingma and Welling, "Auto-encoding variational Bayes", ICLR 2014.
- Goodfellow et al., "Generative adversarial nets", NeurIPS 2014.
- **Ho, Jain and Abbeel, "Denoising diffusion probabilistic models", NeurIPS 2020.**
- Song et al., "Score-based generative modeling through stochastic differential equations", ICLR 2021.
- Rombach, Blattmann, Lorenz, Esser and Ommer, "High-resolution image synthesis with latent diffusion models", CVPR 2022.

**Graphs and geometry**

- Kipf and Welling, "Semi-supervised classification with graph convolutional networks", ICLR 2017.
- Velickovic et al., "Graph attention networks", ICLR 2018.
- Xu, Hu, Leskovec and Jegelka, "How powerful are graph neural networks?", ICLR 2019.
- Bronstein, Bruna, Cohen and Velickovic, "Geometric deep learning: grids, groups, graphs, geodesics, and gauges", 2021 (arXiv:2104.13478).

**Reinforcement learning**

- Mnih et al., "Human-level control through deep reinforcement learning", *Nature*, 2015.
- Schulman, Wolski, Dhariwal, Radford and Klimov, "Proximal policy optimization algorithms", 2017 (arXiv:1707.06347).
- Haarnoja, Zhou, Abbeel and Levine, "Soft actor-critic", ICML 2018.
- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.

**Efficiency and systems**

- Dao, Fu, Ermon, Rudra and Re, "FlashAttention: fast and memory-efficient exact attention with IO-awareness", NeurIPS 2022.
- Rajbhandari, Rasley, Ruwase and He, "ZeRO: memory optimizations toward training trillion parameter models", SC 2020.
- Micikevicius et al., "Mixed precision training", ICLR 2018.

**Trust, safety and theory**

- Goodfellow, Shlens and Szegedy, "Explaining and harnessing adversarial examples", ICLR 2015.
- Madry, Makelov, Schmidt, Tsipras and Vladu, "Towards deep learning models resistant to adversarial attacks", ICLR 2018.
- Guo, Pleiss, Sun and Weinberger, "On calibration of modern neural networks", ICML 2017.
- Abadi et al., "Deep learning with differential privacy", CCS 2016.
- Mitchell et al., "Model cards for model reporting", FAT* 2019.

**Research practice**

- Henderson et al., "Deep reinforcement learning that matters", AAAI 2018.
- Lipton and Steinhardt, "Troubling trends in machine learning scholarship", 2018 (arXiv:1807.03341).
- Pineau et al., "Improving reproducibility in machine learning research", JMLR 2021.
- Bouthillier et al., "Accounting for variance in machine learning benchmarks", MLSys 2021.
- Keshav, "How to read a paper", ACM SIGCOMM Computer Communication Review, 2007.
- Karpathy, "A recipe for training neural networks" (blog post), 2019.

**Books**

- Goodfellow, Bengio and Courville, *Deep Learning*, MIT Press, 2016.
- Prince, *Understanding Deep Learning*, MIT Press, 2023.
- Zhang, Lipton, Li and Smola, *Dive into Deep Learning* (online, d2l.ai).
- Murphy, *Probabilistic Machine Learning: An Introduction* and *Advanced Topics*, MIT Press, 2022 and 2023.

## Reproducing a paper {#reproducing}

Reproducing a result is the best exercise after this course. It forces you to confront every detail that papers omit. A workable process:

1. **Choose well.** Pick a paper with a small-scale core result (a figure you can regenerate in hours on one GPU), public code or at least a clear description, and a clear success criterion. Avoid papers whose headline result needs thousands of GPU-hours.
2. **Define success first.** Write down which numbers or curves you expect and the tolerance (relative to seed variance) before you run anything.
3. **Start from the simplest working version** and verify each component against a known value: shapes, parameter counts (the paper often states them), the initial loss, and a one-batch overfit.
4. **Match the details that matter:** data splits and preprocessing, initialization, optimizer settings, schedule, batch size, evaluation protocol and metric implementation. Differences here explain most failed reproductions.
5. **Use reference code as a check, not a copy.** Read the official implementation after your own attempt to find discrepancies, and cite it. Course policy requires disclosure of external code.
6. **Log everything** with the harness from this page, run several seeds, and keep a "differences from the paper" list.
7. **Report honestly,** including partial success. Discrepancies, such as a metric that moves by 0.5 points because of a data-loading detail, are valuable to the community.

A useful ladder of difficulty: reproduce a figure from a course reading (a day), reproduce a small benchmark number (a week), reproduce an ablation from a recent paper (a few weeks), and extend it with one new experiment (a project).

## Contributing to open source {#open-source}

Open-source contributions are an excellent way to learn how real systems are built and to become visible to future collaborators and employers. Start small and be patient.

- **Pick a library you use,** such as PyTorch, Hugging Face Transformers and Datasets, timm, PyTorch Geometric, Stable-Baselines3, scikit-learn, or this course site.
- **Begin with documentation:** fix typos, clarify an unclear example, add a missing docstring, or translate a page. These are welcome and teach you the workflow.
- **Look for issues labeled "good first issue"** or "help wanted," and comment before you start to avoid duplicated work.
- **Reproduce the bug** first in a minimal script. A minimal, reproducible bug report is itself a contribution.
- **Follow the contribution guide:** coding style, tests, commit message format, and the review process. Keep pull requests small and focused, include a test, and respond to review comments politely and quickly.
- **Add something you built:** a model implementation with a verified result, a tutorial notebook, a benchmark script. Maintainers value correctness and clarity over novelty.
- **Release your project code properly:** a license, a README that works, version pins, and a citation entry.

## Career paths {#careers}

Deep learning work has several overlapping roles. They emphasize different skills; you can move between them, and many people do.

| Path | Typical work | Skills to emphasize | How to prepare |
| --- | --- | --- | --- |
| Research (academic or industrial lab) | Posing new questions, designing experiments, writing papers | Strong math, rigorous experimentation, writing, depth in one area | Reproduce and extend papers, a thesis or project with an advisor, workshops, then conference submissions |
| Applied ML or data science | Turning a business problem into a model with measurable impact | Problem framing, data understanding, baselines, evaluation, communication | Projects on real, messy data; Kaggle-style competitions as practice; domain knowledge |
| ML engineering and MLOps | Training and serving models at scale, pipelines, monitoring | Software engineering, systems, GPUs, distributed training, testing | Strong coding, profiling and deployment practice (Week 16), contributing to open-source infrastructure |
| Specialist roles | Safety and evaluation, interpretability, data and annotation quality, product and policy | Domain-specific rigor, statistics, ethics (Week 17) | Projects that stress-test models; reading and writing evaluations and audits |

General advice: build a small public portfolio of two or three well-documented projects (code, a short write-up, and results with honest error bars), and be able to explain every choice in them. Learn to communicate with non-specialists. Stay curious about the data: most real-world gains come from better data, evaluation and problem framing, rather than a new architecture. Be wary of hype in either direction, and keep checking claims, including your own, against evidence.

## Further courses and resources {#resources}

- **Courses:** Stanford CS231n (convolutional networks for visual recognition), CS224n (NLP with deep learning), CS224W (machine learning with graphs), CS236 (deep generative models); Berkeley CS285 (deep reinforcement learning); MIT 6.S191 (introduction to deep learning); fast.ai Practical Deep Learning; Andrej Karpathy's "Neural Networks: Zero to Hero" video series; the Hugging Face courses and documentation.
- **Fundamentals worth strengthening:** probability and statistics (Murphy; Bishop's *Pattern Recognition and Machine Learning*), optimization (Boyd and Vandenberghe's *Convex Optimization*), linear algebra (Strang), and information theory (Cover and Thomas).
- **Where papers appear:** NeurIPS, ICML and ICLR for general machine learning; CVPR, ICCV and ECCV for vision; ACL and EMNLP for language; plus arXiv and OpenReview (which also hosts peer reviews, an excellent way to learn how papers are judged).
- **Practice:** build things end to end; enter a competition with a time limit; write a blog post explaining one paper; teach this course's material to someone else.
- **Keep your toolchain current:** the specific libraries will change over the years, but the habits in this course (small experiments, baselines, seeds, logs, honesty) will not.

## Further reading {#further-reading}

The reading list above is the reading list for the week. If you read only three things: **Keshav (2007) on reading papers**, **Henderson et al. (2018) on reproducibility and the effect of seeds in deep RL**, and **Karpathy's recipe for training neural networks** (2019).
