# Contributing

Thank you for helping improve this course. Good contributions include fixing typos
or errors, adding exercises, improving explanations, adding diagrams or interactive
demos, and translating pages.

1. Read `SPEC.md`. It defines the page format, callouts, figures, charts, demos, code
   rules and the quality bar.
2. `npm install`, then edit files under `content/`, `figures/` or `demos/`.
3. Run `node build.js --check=N` for the week you changed (renders only that week and
   reports math errors, missing figures or demos, invalid chart JSON and Python syntax
   errors). Run `node build.js` before opening a pull request; it must report zero warnings.
4. Technical claims, numbers and citations must be accurate and verifiable. Cite the
   source for any benchmark number. Write explanations in your own words.
5. Keep diagrams theme-aware (CSS classes, no hard-coded colors) and demos
   self-contained and keyboard-friendly.

By contributing you agree that your text and figures are licensed CC BY 4.0 and your code MIT.
