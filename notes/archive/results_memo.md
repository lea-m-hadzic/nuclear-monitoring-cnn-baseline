# ResNet-50 negative-composition ablation: main results (9 runs)

**Setup.** ResNet-50 is trained only on external layered-fence tiles and evaluated on Saratov, Natanz and Fordow; none of these three AOIs contributes training data.
- **Shared design across conditions:** the same 287 positives and 157 negatives each (126 train / 31 val), and identical optimizer, loss, augmentations and stopping rule. Checkpoints are selected on one shared validation set (118 tiles).
- **Input resolution:** every tile covers 1.9 km. Imagery is extracted at about 0.74 m/px (2560 px) and box-reduced 2x before training, so the **effective model input resolution is about 1.5 m/px** (1280 px). It is not described as "z16 imagery": training tiles come from zoom 16 or 17 sources depending on latitude, resampled to a common ground sample distance.
- **Watermark handling:** a per-tile geometric watermark mask is filled before augmentation.
- **Conditions:**
  - A: nearby negatives only.
  - B: confirmed hard negatives only.
  - C: about 50/50 nearby and hard.
- **Seeds and provenance:** seeds 0, 1 and 2. All 9 runs completed with no technical failures. The inputs were frozen at commit 8742c46 (see `outputs/ablation/freeze.json`).
- **Reporting:** values are mean ± SD over 3 seeds. The facility map is provisional.

## Headline

1. **Hard negatives alone (B) fail to transfer.** Benchmark tile AUROC is at or below chance:
   - pooled 0.55 ± 0.05;
   - Saratov 0.49;
   - Natanz 0.27, which is inverted.

   B also needs the most review area to surface the Natanz and Fordow enrichment plants (43% and 16% of the AOI, against at most 3% for A and C). The likely reason is that a model trained to reject industrial sites learns that "industrial" means negative, and the enrichment plants are industrial sites.

2. **A (nearby) and C (mixed) are tied on tile discrimination:**
   - pooled AUROC 0.76 ± 0.06 vs 0.75 ± 0.05;
   - pooled PR-AUC 0.23 ± 0.13 vs 0.24 ± 0.03.

   C is about 4× more stable across seeds on PR-AUC.

3. **C finds Saratov facilities with less review area.** Saratov has 24 of the 26 known facilities. With 20% of the AOI reviewed, Saratov facility recall is:
   - C 0.54 ± 0.04;
   - A 0.32 ± 0.27;
   - B 0.19;
   - random 0.24.

   The area needed to find half the Saratov facilities is 18.5 ± 1.3% for C against 33.6 ± 21.2% for A. C surfaces all three warhead-storage sites within about 4% of the AOI.

4. **The single enrichment plants favour A.** Natanz is found after 0.15% of the AOI for A and 1.7 ± 2.8% for C; one C seed misses it at a 1–2% budget. Fordow is similar for both: 2.1% for A, 2.8% for C. These are single-facility AOIs, so their recall is binary and noisy.

5. **AOI-specific reversals.** The pooled ranking (C > A > B) is driven by Saratov. At Natanz, and at Fordow up to a 5% budget, A is ahead of C. On Fordow tile AUROC the order is C > A. All reversals are listed in `aoi_reversals.csv`. At Fordow's 10% budget, A and C are tied at 1.0.

## Caveats (state these with the results)

- **The shared validation set favours C by construction.** Its negatives are half nearby and half hard, the same mixture as C's training negatives. C's higher validation score (PR-AUC 0.94 vs 0.72 for A and 0.79 for B) is therefore consistent with the benchmark ranking, but it is not independent evidence for C: the validation distribution suits C. Only the benchmark results compare the conditions on neutral ground.
- **A stops very early.** Its best epoch is 0–3, and its validation loss swings widely, so A is effectively a lightly fine-tuned model.
- **Three seeds is small.** Seed ranges are 0.09–0.10 in pooled tile AUROC and up to 0.27 SD in Saratov recall. I would make no significance claims, and describe A vs C as comparable on discrimination and different in where they find facilities.
- **The tile label and facility credit can disagree.** On Saratov tiles A has the higher tile AUROC (0.77 vs 0.73), yet C has better facility recall. The tile label (`human_target`) and the footprint-based facility credit are measuring different things.
- **The facility map is provisional.** Natanz and Fordow have one facility each. All conditions need 87–92% of the AOI reviewed to find every Saratov facility.
- **Positive-label QA (completed after these runs).** All 287 positives were reviewed visually. The 14 flagged ones were reviewed by Yanliang and confirmed valid, so no positive was changed (`outputs/qa/label_qa_report.md`). A targeted review of 41 negatives is with Yanliang. No negative label has been changed, and the effect of any contamination it finds is not yet known. The C_matched and unmasked runs are reported in `results_robustness/robustness_memo.md`.
- **Finer resolution is untested.** All results are at about 1.5 m/px effective input. Whether finer input (for example native z17, about 0.6 m/px) would improve retrieval is unknown. A proposed optional sensitivity is specified but not run (`outputs/ablation/z17_sensitivity_spec.md`).

## Interpretation for the paper

Negative composition matters mainly for **where** the model finds facilities, not for overall tile ranking.
- Hard negatives on their own teach a shortcut that suppresses real nuclear industrial sites.
- Nearby negatives give the sharpest single-site hits (Natanz).
- Mixing the two gives the most consistent and most budget-efficient retrieval across the many Saratov sites, and the lowest seed variance.

The recommended default is C, with the Natanz reversal reported. This choice rests on the benchmark comparison above. The shared-validation result agrees with it, but because that set has C's mixture it counts as supporting, not independent, evidence. The paper should say the mixed design was chosen from this ablation.

## Files (outputs/ablation/results/)

| Table | Content | File |
|---|---|---|
| T1 | Checkpoint selection | `T1_checkpoint_selection.csv` |
| T2 | Tile discrimination | `T2_tile_discrimination.csv` |
| T3 | Facility recall at equal review budget, with random reference | `T3_facility_recall_at_budget.csv` |
| T4 | Review area to reach a recall target | `T4_review_area_for_recall_targets.csv` |
| T5 | Review area to surface each facility | `T5_per_facility_review_area.csv` |
| — | All tables in Markdown | `results_tables.md` |

| Figure | File |
|---|---|
| Facility recall vs review budget | `fig_facility_recall_vs_budget.png` |
| Tile metrics per AOI, with seeds | `fig_tile_metrics.png` |
| Per-facility first-hit review area | `fig_per_facility_first_hit.png` |
| Shared-val training curves | `fig_training_curves.png` |

Per-run long-format data: `runs_long.csv`, `facility_recall_at_budget_long.csv`, `facility_first_hits_long.csv`, `training_curves_long.csv`, `seed_ranges.csv`.
