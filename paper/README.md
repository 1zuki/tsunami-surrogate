# Paper folder

This folder contains a clean manuscript scaffold aligned with the codebase.

## Main files

- `main.tex`: paper entry point
- `sections/`: modular section files
- `references.bib`: starter bibliography
- `figs/`: exported paper figures
- `Makefile`: basic compile commands

## Build

```bash
cd paper
make
```

If `latexmk` is unavailable on your machine, you can also run:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Writing strategy

A good first draft for this project usually follows this order:

1. fill the abstract after experiments are done
2. finish Methods and Experimental Setup next
3. write Results from figures and tables
4. polish Introduction and Conclusion last
