# UEMS 2026 Presentation

Beamer slides (15 frames, 16:9) for the 1st UEMS Congress 2026 (Leuven, 27–30 May 2026).

## Build

```bash
cd presentation
make           # produces main.pdf
make clean     # remove aux files and the PDF
make view      # open the PDF
```

Requires TeX Live with Beamer + TikZ + babel (already available on this VM).

## Structure

- `main.tex` — root document, theme, title metadata.
- `sections/01-title.tex` … `15-conclusion.tex` — one frame per file, edit independently.
- `figures/` — PNGs copied from `Datasets/Out/`.

## Editing tips

- Each slide is self-contained; reorder by editing the `\input{}` list in `main.tex`.
- Color palette: `medBlue`, `medTeal`, `medAccent`, `medSoft` (defined in `main.tex`).
- To switch to Spanish: change `\usepackage[english]{babel}` to `[spanish]` and translate text in `sections/`.
