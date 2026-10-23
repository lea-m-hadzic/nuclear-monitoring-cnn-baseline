# Targeted negative-label review: status record

**Status:** sent to Yanliang on 2026-10-21; **review pending. No negative label has been changed.**

**What was sent:** 41 of the 382 negatives used by the frozen ablation manifests (A, B, C, C_matched): 38 nearby and 3 confirmed hard negatives. Split: 35 train, 6 val.

**Selection:** existing metadata and QA heuristics only, fixed before looking at which tiles they pick. **No model scores were used.**

| Rule | Tiles | Criterion |
|---|---|---|
| P | 23 | Nearby negative in the closest 10% to its parent positive's fence point (≤ 3.25 km) |
| X | 5 | Parent belongs to a multi-point complex (≥ 3 positive points) and the tile is closer than the median (4.43 km) |
| B | 12 | Top 5% edge density among nearby negatives (≥ 0.239): built-up or industrial scene |
| H | 3 | Confirmed hard negative flagged at pre-extraction screening or earlier review |

Two tiles meet two rules each. Requested answer per tile: `keep_negative` / `contains_layered_fence` / `unclear`.

**Files:** the KML (Google Earth) and CSV contain **coordinates** and are kept local only, in `external_imagery/dataset/review_packages/negative_review_targeted/` (git-ignored). Generator: `cnn_benchmark/external_data/review_package_negatives.py`.
