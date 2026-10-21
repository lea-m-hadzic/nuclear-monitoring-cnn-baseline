# Watermark / mosaic-artifact audit (non-destructive)

Produced by `python src/artifact_audit.py --n 120 --supplementary-pairwise`: 120 random resolved parent tiles per AOI
at the 1280 px model input. **Refreshed 2026-10-09 with the complete canonical Natanz set** (sampled from 776 tiles,
previously 246); the method, seed and thresholds are unchanged. **No preprocessing has been applied to the
pipeline.**

## What differs by AOI

| | Saratov z16 | Fordow | Natanz |
|---|---|---|---|
| Source imagery | zoom-16, 2×2 source images per tile | zoom-17, 4×4 mosaic, cropped and resampled | zoom-17, 4×4 mosaic, cropped and resampled |
| Watermark rows ("Google" + attribution strip), y at 1280 px | ≈ 640, ≈ 1270 (2 rows) | ≈ 206, 629, 1052 (3 rows × 3) | ≈ 415, 843, ≈ 1270 (3 rows, different offset) |
| Fixed-pattern pixels (contrast in > 50% of tiles) | 1.1 % | 1.7 % | 1.6 % (1.58; was 1.56) |
| Watermark mask area (dilated) | 1.7 % | 2.5 % | 2.3 % (2.29; was 2.27) |
| Column colour seams (gradient ratio at seam) | weak, ≈ 1.1–1.26 at x ≈ 640 | weak, ≈ 1.1–1.2 | weak, ≈ 1.1 |

The strong peaks in the row-gradient profile (ratio 2.4–4.3) coincide with the watermark rows (see
`fixed_pattern_maps.png` and `seam_profiles.png`). Genuine colour/brightness seams between source images exist
but are weak on average.

## How identifying are they?

AUC for separating Saratov from Fordow using artifact features only (0.5 = no information):

| Feature | AUC | Separation (2·\|AUC−0.5\|) |
|---|---|---|
| Watermark contrast at Saratov positions | 1.000 | 1.00 |
| Watermark contrast at Fordow positions | 0.008 | 0.98 |
| Column seam strength (Saratov grid / Fordow grid) | 0.946 / 0.069 | 0.89 / 0.86 |
| Row "seam" strength (mostly watermark rows) | 0.707 / 0.037 | 0.41 / 0.93 |
| **Watermark contrast after union-mask fill (Saratov / Fordow positions)** | **0.549 / 0.541** | **0.10 / 0.08** |

The Saratov-vs-Fordow table is identical to the 246-tile audit: the same Saratov and Fordow tiles are sampled, and
the union mask changed by only 0.01 percentage points. A trivial feature tells the AOIs apart perfectly. Within one AOI the pattern is constant, so it cannot help
Saratov spatial CV. It is a real confound for leave-one-AOI-out (a domain-shift signal) and for any model trained
on mixed AOIs (region identity correlates with base rates). Even Fordow and Natanz differ in watermark row
position. Note: the column-seam AUCs in the table are measured at the AOI-specific seam positions, so they may
also carry landscape differences.

## All AOI pairs, including Natanz (supplementary, 2026-10-09)

Separation (2·|AUC−0.5|) of watermark-contrast features for each AOI pair, before and after the same union-mask
fill (`supplementary_pairwise_separation.csv`):

| Pair | Before (min–max over the 3 position features) | After union-mask fill |
|---|---|---|
| Saratov vs Fordow | 0.98–1.00 | 0.08–0.10 |
| Saratov vs Natanz | 1.00 | 0.20–0.23 |
| Fordow vs Natanz | 0.54 (Saratov positions), 1.00 (Fordow/Natanz positions) | 0.13 |

- **Natanz is identifiable from the watermarks** relative to *both* other AOIs, including Fordow (another zoom-17
  mosaic), because its watermark rows sit at different heights (y ≈ 415 / 843 vs 206 / 629 / 1052).
- **The shared mask removes most of this** for every pair. The Natanz pairs keep more residual separation
  (0.13–0.23) than Saratov vs Fordow (about 0.1).
- **Caveat:** each feature is "contrast inside the mask minus contrast outside it". After the blur fill, the inside
  term is close to zero, so the residual largely reflects background texture (rugged desert vs farmland), not the
  watermark text. This metric cannot cleanly separate the two.

## Changes vs the 246-tile audit (2026-10-08)

- **Unchanged:** the Saratov/Fordow samples, every Saratov-vs-Fordow AUC (to three decimals), the mask component
  counts, and the Saratov/Fordow seam and watermark-row positions.
- **Natanz, now sampled from all 776 tiles (18 of 120 tiles overlap the old sample):**
  - watermark rows are confirmed at y ≈ 415 and 843 (gradient ratio ≈ 2.9, previously ≈ 3.1);
  - minor peaks from the partial sample (y ≈ 321, 613–673) disappear and look like sampling noise;
  - mask area 2.27 → 2.29 %, and the union mask 5.81 → 5.82 %.
- **New:** pairwise separation that includes Natanz (above). Previously only Saratov vs Fordow was quantified.
- **Conclusions are unchanged in substance:** the watermark layout identifies the AOI, and one shared mask is the
  only simple option that removes most of the signal. **New nuance:** a small residual remains for the
  Natanz pairs, so leave-one-AOI-out with Natanz should state this as a limitation.

## Preprocessing options (not applied)

| Option | Removes region signal? | Effect on fence visibility | Verdict |
|---|---|---|---|
| Edge cropping | No: watermark rows are inside the tile (e.g. y ≈ 206 / 629 / 1052) | Loses context at the edges | Not useful |
| Standardised resizing (1280 / 640) | No: the watermarks persist at every scale | 640 loses fences | Not useful for artifacts |
| AOI-specific watermark masking | Removes the text, but leaves blurred strips at region-specific positions, which is itself a giveaway | ≈ 2 % of pixels | Not recommended |
| **Union watermark mask, identical for every tile** (fill by local blur) | **Mostly: separation 0.98–1.00 → 0.08–0.10 (Saratov/Fordow) and → 0.13–0.23 for the Natanz pairs** | 5.8 % of pixels, in thin horizontal strips ≈ 15–30 m tall; a fence crossing a strip loses one short segment, enclosures remain | **Recommended candidate** |
| Random translation / crop augmentation | Partly (positions only); text appearance remains | None | Optional complement |

Not addressed: the sharpness difference between zoom-16 and zoom-17 sources. It is confounded with landscape and
can't be removed simply; report it as a caveat for leave-one-AOI-out.

## Recommendation for the meeting

1. Keep Saratov spatial CV, the primary within-region result, **unmasked or masked**; artifacts are constant
   within an AOI. For consistency, use the same preprocessing in every run.
2. For **leave-one-AOI-out and any mixed-AOI training**, use the **union watermark mask** (same fixed mask for
   all tiles, blur-filled). State as limitations the residual source-sharpness difference and the small residual
   separation for the Natanz pairs (0.13–0.23 after masking, likely mostly terrain).
3. Decide whether to apply the mask everywhere (simplest; one pipeline) or only for leave-one-AOI-out.
   Implementation is a deterministic transform in the dataset (≈ 20 lines) with the mask saved as a versioned
   file; it should be added only after the meeting decision.
4. Before adopting it, check that no labelled facility perimeter lies mostly under a watermark strip
   (quick audit over the 70 target-facility tiles).
