# Poster (DML course, A2)

`poster.tex` on the course template (`beamerposter.sty`, `beamerthemeZH.sty`, `img/`: leave
the style and fonts as they are). `poster.pdf` is the current build.

## Build

- **Overleaf:** upload this whole folder and compile `poster.tex` with pdfLaTeX (it runs
  BibTeX itself).
- **Windows:** install MiKTeX, then `powershell -ExecutionPolicy Bypass -File poster\build.ps1`.
- Before submitting, check that the PDF is 420 x 594 mm (A2) and fits on one page.

## What comes from where

| File | Made by | Edit |
|---|---|---|
| `fig/results.tex` | `make_figures.py`: every number in the text (win rates, self-destructs, KL, ...) | regenerate, don't type numbers into `poster.tex` |
| `fig/rl_curves.pdf`, `fig/style.pdf`, `fig/imitation_loss.pdf` | `make_figures.py`, from the training logs in `runs/` (not in git) | regenerate |
| `fig/hero.jpg`, `fig/combo_strip.jpg`, `fig/combo_strip.tex` | `capture_combos.py` (screenshots of the bot's combos), cropped by `make_figures.py` (`HERO`, `STRIP`) | regenerate |
| everything else in `poster.tex`, `bib.bib` | by hand | yes |

`make_figures.py` and `capture_combos.py` need the training runs and Dolphin on the machine
that trained the imitation and RL models. Without them the committed figures are used as they are.

## Still to fill in

- Names: `\author{...}` in `poster.tex`.
- E-mail in the footer: `\href{mailto:...}` in `beamerthemeZH.sty` (the one content line in the theme).

## Adding the pure-RL models

The poster tells one story: pure RL (spam), then RL with a reaction delay, then imitation and
imitation + RL (this repository). Results of the RL models go into the same blocks:

- Win rates: add tiles with `\stattile{<win %>}{<opponent>}{<model>}` (defined in the preamble).
- Comparison table or figure: keep the same opponents (level 9 CPU on Final Destination,
  4 stocks) and, if possible, the same style metrics as `fig/style.pdf`, so the four models
  compare directly.
- Space is tight: both columns are full, so trim text when adding a block.
