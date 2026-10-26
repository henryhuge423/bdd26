# stage report 1

English, Chinese and Chinese-reader report PDFs and LaTeX sources.

## Delivery layout (2026-10-26)

This directory contains the report PDFs, LaTeX sources and every referenced figure/input, plus build instructions. Verification JSON, figure CSVs, unused raster previews and montage source tiles were moved to the project owner's ignored operations archive; historical verification descriptions above refer to those archived records. The report PDFs and TeX content are unchanged by this cleanup. Research evidence and current outcomes are indexed in [the project results](../../results/README.md).

Build the English source with `pdflatex` and Chinese sources with `xelatex`, twice from this directory, unless the supplied build script specifies otherwise. Keep temporary compiler output in a separate build directory. No datasets or checkpoints are required to compile these reports.
