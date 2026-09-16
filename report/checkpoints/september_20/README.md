# September 20 literature-survey checkpoint

This checkpoint is a thin LaTeX wrapper around the report's authoritative prose. It does not maintain a second literature survey or research-gap copy:

- the problem statement inputs `sections/shared/introduction_problem_statement.tex`;
- the literature survey inputs `sections/02_literature_survey.tex`;
- the research gap and objectives input `sections/03_research_gap_objectives.tex`;
- references use `references/AMP_VERIFIED_REFERENCES.bib`.

It intentionally excludes the full AMP-LDP methodology, implementation, results, limitations, and conclusion sections.

Build from the `report/` directory:

```text
pdflatex checkpoints/september_20/checkpoint.tex
bibtex checkpoint
pdflatex checkpoints/september_20/checkpoint.tex
pdflatex checkpoints/september_20/checkpoint.tex
```

The Stage 31 host has no TeX engine or BibTeX executable installed, so the checkpoint was validated statically rather than compiled.
