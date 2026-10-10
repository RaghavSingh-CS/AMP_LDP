# AMP October 10 Methodology Checkpoint

This is the independent Stage 37 coursework submission. It preserves the submitted September 20 problem statement, literature survey, research gap, objectives, citations, and comparison table, then adds Section 4, Methodology.

## Source identity

The September body is copied from the authoritative `report/checkpoints/september_20/` dependency set. The title-block email addresses are taken from the submitted `AMP_LDP_Literature_Survey.pdf`, because those addresses are present in the submitted PDF but not in the repository's September source.

The methodology is a concise restatement of the frozen implemented procedure documented in `docs/paper/AMP_METHOD_FREEZE.md`, `docs/preprocessing_decisions.md`, `report/sections/04_methodology.tex`, and the committed Stage 14, 16, 17A, 17B, 24A, 25A, 25B, 27A, and 27B protocol artifacts. It contains no experimental results or conclusions.

## Structure

- `main.tex`: Overleaf entry point.
- `sections/02_literature_survey.tex`: preserved literature survey wrapper.
- `sections/03_research_gap_objectives.tex`: preserved research gap and objectives wrapper.
- `sections/shared/`: preserved September source sections.
- `sections/04_methodology.tex`: new methodology checkpoint section.
- `figures/amp_ldp_architecture.tex`: pipeline and matched-control diagram.
- `references/AMP_VERIFIED_REFERENCES.bib`: verified 24-paper bibliography.
- `sn-jnl.cls`, `sn-mathphys-num.bst`: Springer Nature dependencies.

## Build

Set `main.tex` as the main document and compile with pdfLaTeX and BibTeX:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Validation status

Static validation on 10 October 2026 found no missing inputs, citation-key
mismatches, unresolved internal references, duplicate labels, unbalanced LaTeX
environments, or absent figure files. No Windows or Ubuntu WSL LaTeX engine or
BibTeX executable was available, so the checkpoint was not compiled locally.
Final pagination and visual overflow checks must be completed in Overleaf.
