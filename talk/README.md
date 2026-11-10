# Study the animal, not the solver

Talk, Gabriel Weymouth (TU Delft): WaterLily for biological flows. A plain [reveal.js](https://revealjs.com) deck, vendored, no build step.

## Run it
Needs Julia 1.10 or later. All commands run from the repo root.
```
julia --project=talk -e "using Pkg; Pkg.instantiate()"    # first time only
julia --project=talk --threads=auto talk/server.jl        # serves the deck on localhost:8081 and opens it
```
Without Julia, `python talk/scripts/serve.py` serves the deck on localhost:8000.

Keys: Esc overview · S speaker notes · L light/dark · H hide placeholders · F fullscreen.

## Whale demo (slide 4)
```
julia --project=talk/demo -e "using Pkg; Pkg.instantiate()"   # first time only
julia --project=talk/demo -i talk/demo/whale.jl
```
Call `show_whale()` once before the talk to compile, then `show_whale(Λ)` with a sweep angle from the room (0–35°): about 11–13 s per stroke at 25° and 13–15 s at 35° on the A2000 (LilyPad semi-Lagrangian step, NarrowBand, Biot–Savart box).

## Editing
- `index.html` holds every slide in order, numbered from 0 as presented: headline, media, and speaker notes in `<aside class="notes">`. Elements with `class="fragment"` step in on a click; a shared `data-fragment-index` brings them in together.
- `js/figures.js` draws the coded graphics named by `data-figure`.
- After changing `css/` or `js/`, bump the `?v=` tags in `index.html`: browsers cache them.
