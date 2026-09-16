# AMP final report foundation

This directory contains the Stage 30 LaTeX foundation for the 2026–27 ML Lab final report, provisionally titled **“Weakly Supervised Latent Distribution Profiling for Customer-Level Electricity Theft Detection.”** It uses the official Springer Nature `sn-jnl` template and the numbered mathematics/physics bibliography style.

## Authoritative sources

The Markdown files in [`../docs/paper/`](../docs/paper/) are authoritative for experimental chronology, method definitions, numerical results, disclosures, limitations, and allowed claims. The Stage 29 literature package in [`../docs/paper/literature/`](../docs/paper/literature/) is authoritative for references and literature comparisons. Do not infer or recompute experimental results while drafting the report.

`references/AMP_VERIFIED_REFERENCES.bib` is a byte-for-byte Stage 30 copy of `../docs/paper/literature/AMP_VERIFIED_REFERENCES.bib`. It contains exactly 24 verified journal references. Do not add a reference directly to the report copy unless the authoritative Stage 29 bibliography has first been updated through a separately governed literature-verification stage.

## Structure

- `main.tex`: Springer manuscript entry point and section ordering.
- `cover.tex`: faculty-facing cover page, separate from the Springer title block.
- `sections/`: modular placeholder files for later manuscript drafting.
- `references/`: verified Stage 29 BibTeX copy.
- `sn-jnl.cls` and `sn-mathphys-num.bst`: unmodified files from the official December 2024 Springer Nature template package.
- `SPRINGER_TEMPLATE_SOURCE.md`: provenance and checksums for the downloaded template.

The official Springer template recommends a single `.tex` file for journal submission. Modularity is retained here for the faculty report drafting workflow; flatten the source only if a later submission workflow requires it.

## Build

From this directory, with a compatible TeX distribution installed:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The current Stage 30 host does not have `pdflatex`, `latexmk`, `bibtex`, or `biber` available, so no preliminary PDF was generated. A TeX distribution was not installed because that would be a large environment change requiring explicit approval.

## Information still required

- Faculty guide name and any required designation.
- Verified MIT/School logo asset; if supplied, save it as `assets/mit-logo.png`.
- Author email addresses.
- Roll number and registration number for each author.
- Any faculty-mandated wording beyond the supplied school, institution, title, and academic year.

All missing fields are displayed as `TO BE PROVIDED`; no identity detail has been invented.
