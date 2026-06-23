# CLAUDE.md — Project conventions

## UI stack
- **Tailwind CSS** (Play CDN) for all layout and styling
- **Preline UI** (CDN) for interactive components — note: Preline's overlay variants (`hs-overlay-open:*`) require the Tailwind plugin and a build step, so they don't work with the Play CDN. Use Dash callbacks to toggle `className` between `hidden …` and `flex …` for modals instead.
- **No `dash-bootstrap-components`** — removed; do not re-add it
- **No custom CSS files** unless a Dash component's internals cannot be reached with Tailwind (e.g. `dcc.Dropdown` internal colours use inline `style=` props instead)

## UI principles
- Keep the interface simple and dense — this is a technical tool, not a product page
- Dark theme throughout: `bg-[#0d0d0d]` root, `bg-neutral-900` panels/modals, `text-gray-200` primary text
- Prefer Tailwind utility classes in `className`; avoid inline `style=` except for dynamic values (e.g. progress bar width, Plotly figure colours)
- Markdown content rendered in modals: use `prose prose-invert prose-sm max-w-none` — no separate CSS file needed

## Architecture notes
- `app.py` — minimal entry point only; no Dash/Plotly imports (prevents Windows spawn re-import in worker processes)
- `webapp.py` — all Dash layout, callbacks, and computation state
- `physics/` — CR3BP integrator, coordinate transforms, optimiser (pure NumPy/SciPy, no UI imports)
- `destinations/` — plugin pattern; add new targets by subclassing `Destination`
- Results cached in `cache/<dest_id>.npz`; delete the file to force recomputation
