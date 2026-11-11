# Stage report 7

The English report, detailed Chinese report, and Chinese reader cover P17–P26, with quantitative comparisons and an object-level casebook. The images show fixed last-checkpoint examples. `NUMBER_SOURCES.md` identifies the measurements and source files. `evidence/casebook.json` records GT image/object IDs, coordinates, areas, local responses, and input hashes.

Build on Windows with TeX Live 2022, CTeX, PGFPlots and Windows Chinese fonts: `powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1`. The script runs XeLaTeX three times for each report and checks layout warnings. Render all pages for visual review. `python verify_numbers.py` checks the P25 aggregates, sensitivity record and GPU-time arithmetic. `python make_casebook.py` rebuilds the cropped and enlarged case images from the included originals using Pillow.
