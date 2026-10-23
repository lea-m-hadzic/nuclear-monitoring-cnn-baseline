# CNN baseline for "Catching Proliferation at the Fence": status for the author meeting

*Branch `cnn-benchmark-infrastructure`. Details are in `cnn_benchmark/README.md`.*

## Goal

Answer the reviewer's request for a representative supervised CNN baseline **without reintroducing tile-level
metrics as the headline**. Both the CNN and the VLM are judged on the paper's operational question: *how much
imagery must an analyst review to surface the known facilities?*

## What is implemented

- **Canonical z16 parent-tile table** (Saratov 1,387 / Natanz 776 / Fordow 777) built from the master
  workbook. The human target (layered-fencing signature = TP or FN) comes from labels only, never from VLM scores;
  quadrant labels are aggregated to the parent and conflicts are flagged, not fixed.
- **Provisional facility map** reconstructed from label text: 24 Saratov targets, plus Natanz and Fordo; a
  candidate 21st silo and the waste repository are kept but outside the default denominator.
- **Model-agnostic facility evaluator** (any `tile_id, score` file): recall vs. km² and % of AOI reviewed, area
  needed for 50/75/90/100% recall, per-facility first-hit rank and area, workload reduction, with an explicit
  tie policy.
- **Candidate leakage-safe splits** with automatic checks: Saratov 5-block spatial CV with a 1-tile buffer, and
  leave-one-AOI-out.
- **Full-resolution pipeline:** cache resolver (extracted folders and zip parts, matched on exact filename), an
  integrity report, and ResNet-50 training/inference. Training is blocked until every benchmark parent resolves.
- **Tests:** 66 pass (0 skipped, 0 expected failures).

## Manuscript numbers reproduced: 96 / 97 checks

- **Table 2:** facilities surfaced (9 → 13 at z16, 11 → 18 at z17, matching by facility type), flagged areas
  (216.6 / 280 / 130.9 / 160.4 km²), percentages, API calls.
- **Table 3:** all 7 threshold rows. **Table 4:** Natanz and Fordo surfacing and areas.
- **Results-text counts** also match.
- **The one miss:** the text says all six missed z17 silos were cut off at tile edges, but `SAR-SILO-11` has no
  "cut off" label.

## Data completeness

| AOI | Parent tiles with full-res image | Human-positive parents | Known target facilities |
|---|---|---|---|
| Saratov | 1,387 / 1,387 | 102 | 24 (+1 candidate) |
| Fordow | 777 / 777 | 36 | 1 |
| Natanz | 776 / 776 | 8 | 1 |

All 2,940 images are 2560×2560 RGB PNGs at 0.742 m/px, and all decode without errors. **Natanz imagery: RESOLVED
(2026-10-09).** Qiqi's canonical re-upload contains all 776 tiles. The earlier shared folder held only 246 of them,
plus 261 off-grid tiles; the 246 overlapping tiles are byte-identical, and the old folder is no longer used. The
decoded 1280 px cache is complete (2,940 tiles, 14.45 GB), and dry runs pass for Saratov CV and for both Iranian
leave-one-AOI-out folds.

## Proposed design

- **Training target:** a z16 parent tile contains the human-recognised layered-fencing signature (TP or FN,
  aggregated from quadrants). This matches the VLM prompt and gives more positives than training on nuclear
  sites only.
- **Model:** ImageNet-pretrained ResNet-50 with weighted BCE and rot90/flip augmentation. Input is 1280 px
  (2× box reduction): double fences stay separable at 1280 px but merge at 640 px.
- **Evaluation (headline):** rank all parent tiles by score; plot facility recall vs. area reviewed, using the
  **same parent-footprint rule for CNN and VLM**. Secondary: area for 50/75/90% recall; per-facility first-hit
  area (the only meaningful metric for Natanz and Fordo, which have one facility each); workload reduction.
  The manuscript's own reasoning-validated rule is reproduced separately and never mixed in.
- **Splits (candidates):**
  - Saratov spatial CV: within-region, out-of-fold scores for all 1,387 tiles; the facility recall curve over 24
    targets.
  - Leave-one-AOI-out: the cross-region transfer story; Iran results are n = 1 anecdotes.
- **Framing:** the VLM is zero-shot (0 labels); the CNN is supervised. Always report the number of labelled tiles,
  and never present within-region CV as generalisation.

## Unresolved methodological choices (to decide)

1. **Imaging artifacts:** whether to mask watermarks before leave-one-AOI-out (see the artifact audit below).
2. **Tie handling** in the headline VLM curve: the conservative "whole tie group" default vs. the expected-rank
   variant.
3. **Natanz in leave-one-AOI-out:** with only 8 positive parents, whether Natanz is used for training at all.
4. **Ambiguous labels** (cut-off, renovation, borderline "fence + track"): keep in training, or run a
   sensitivity analysis.
5. **Training schedule:** a fixed 20 epochs with no tuning, or early stopping on spatial validation.
6. **Whether to add the z17 Saratov comparison** (appendix only, never mixed with z16).

## Questions for Yanliang / Qiqi

**Qiqi:** the Natanz imagery is resolved, thank you. Please confirm that `natanz_30km_tiles_auto_mosaic_reupload`
is the folder the VLM run actually used (all filenames and the 246 overlapping files match byte-for-byte).

**Yanliang (must decide before final training):** see Part A of `outputs/facility_mapping_questions.md`:

1. Is there an authoritative facility list?
2. Is the 21st silo in or out? (It sits about 40.0 km from the centre, on the AOI boundary.)
3. What is Fordo's footprint?
4. Natanz footprint and the 4 conflicting parent labels.
5. How were reasoning-disqualified tiles labelled?
6. Should footprints include partly visible facilities?

Part B (names, renovation silos, text corrections) can wait.

## Full-resolution sanity check (Saratov fold 0, 3 epochs; `FULLRES_SANITY_CHECK_ONLY`)

- **Run:** pretrained ResNet-50 at 1280 px, batch 4, on the M5 Max (MPS). 33.5 min total; about 700 s per
  epoch; peak GPU memory 13.7 GB.
- **Loss:** training loss falls from about 1.18 to about 0.7–0.9 (noisy); weighted validation loss falls
  2.54 → 1.68.
- **Outputs:** checkpoints reload, test predictions export one score per tile, and standalone scoring works.
  Accuracy is not interpreted. Details: `outputs/fullres_sanity_check_summary.md`.
- **Runtime:** CPU-side PNG decoding competes with the GPU on this chip. With the decoded 1280 px cache (now
  built for all 2,940 tiles), training runs at 0.33 s/img. The full 8-run benchmark is about 16 h on the M5 Max,
  or roughly 2–3 GPU-hours on an A100.

## Imaging-artifact audit (`outputs/artifact_audit/artifact_audit.md`)

- **The watermark grids identify the region.** Saratov has 2 rows of watermarks; Fordow and Natanz have 3 rows
  each, but at different heights. Watermark contrast alone separates every AOI pair (separation about 1.0,
  Fordow vs Natanz included); colour seams are weak. Refreshed with all 776 Natanz tiles; the conclusions are
  unchanged.
- **A simple fix mostly works.** One union mask applied identically to every tile (5.8% of pixels, thin
  horizontal strips, blur-filled) drops Saratov-vs-Fordow separation to near chance (about 0.1). For the Natanz
  pairs it falls to about 0.13–0.23, a residual that probably reflects terrain texture rather than watermarks.
  Enclosures stay visible. Edge cropping and resizing don't help, and per-AOI masks would leave region-specific
  blur strips.
- **Proposal:** decide at the meeting whether to apply the union mask to every run (simplest) or only to
  leave-one-AOI-out. Saratov spatial CV is unaffected either way, because the pattern is constant within an AOI.
  The sharpness difference between zoom-16 and zoom-17 sources remains a stated limitation.

