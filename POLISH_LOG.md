# Polish Log

## Iteration 1

- **Footer contrast fix** — `index.html`: footer `color` raised from `#374151` (1.8:1 contrast) to `#6b7280` so it's actually legible on the `#0f0f13` background.
- **Hero note contrast fix** — `index.html`: `.hero-note` `color` raised from `#4b5563` to `#6b7280` for minimum readable contrast at 13px.
- **Activity card remove button tap target** — `build.html`: increased `min-height` from 24px → 36px, added `min-width: 36px` and flexbox centering so the "×" is comfortably tappable on mobile.

## Iteration 2

- **Build page low-contrast text** — `build.html`: `.step-eyebrow`, `.hint`, `.privacy-note`, `.dot-label` all used `#4b5563` on dark card backgrounds (≈2.5:1 contrast). Raised to `#6b7280` (≈4.3:1).
- **Result page "Edit Info" tap target** — `result.html`: `.back` button had no min-height; added `min-height: 36px; display: inline-flex; align-items: center` so it meets minimum touch target on mobile.
- **Template picker card description font** — `result.html`: `.tmpl-card-desc` bumped from `11px` → `12px` for better readability in the template picker UI.

## Iteration 3

- **Back button hover nearly invisible** — `build.html`: `.back-btn:hover` border changed from `#4b5563` → `#6b7280` so the hover state is actually distinguishable from the resting `#2d2d3d` border on the dark theme.
- **Back link/button hover transitions missing** — `build.html` and `result.html`: added `transition: color 0.15s` to `.back` so the color shift on hover is smooth rather than an instant snap.
- **Loading status text transition** — `build.html`: added `transition: opacity 0.3s` to `.loading-status` and updated `showLoadingOverlay` to fade the element out before swapping text, then fade back in — eliminates the jarring text-snap during resume generation.
