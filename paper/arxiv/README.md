# arXiv wrapper

This directory contains a neutral `article`-class wrapper for the same
manuscript used by the Elsevier `cas-sc` submission. It reuses the canonical
files in `paper/sections/`, `paper/figures/`, and
`paper/sections/references.bib`; scientific text should not be edited in a
separate arXiv copy.

## Build locally

Run from the `paper/` directory so the shared paths resolve exactly as they do
in the journal manuscript:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=arxiv/build arxiv/main.tex
```

The resulting PDF is:

```text
paper/arxiv/build/main.pdf
```

Local build output is covered by the repository's existing ignore rules.
Generated source archives are upload artifacts and should remain uncommitted.

## Prepare a source archive

Create a clean staging directory whose top-level file is `main.tex`. Include
the shared sections, bibliography, and figures, but do not include the
Elsevier class files, journal letters, UIT forms, build directories, datasets,
checkpoints, or evaluation outputs.

From the repository root:

```bash
stage="$(mktemp -d)"
cp paper/arxiv/main.tex "$stage/main.tex"
cp -R paper/sections "$stage/sections"
mkdir -p "$stage/figures"
cp paper/figures/*.pdf "$stage/figures/"

(
  cd "$stage"
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
  rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.log main.out
  tar -czf tsunami-surrogate-arxiv-source.tar.gz \
    main.tex sections figures
)

cp "$stage/tsunami-surrogate-arxiv-source.tar.gz" paper/arxiv/
rm -rf "$stage"
```

Compile the staged source successfully before uploading it. The wrapper omits
Elsevier-only highlights and submission metadata but otherwise uses the same
manuscript sections, figures, citations, declarations, and availability text.
