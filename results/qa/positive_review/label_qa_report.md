# Label QA: external positives and the provisional benchmark facility map

This is a read-only check: no label, manifest or frozen result was changed. Flagged positives are listed in `positive_proposed_corrections.csv`. **Update 2026-10-21:** Yanliang reviewed all 14 and confirmed them valid; see §1a.

- **Tool:** `external_data/label_qa.py`.
- **Private material:** visual sheets and per-positive tables are in the git-ignored `external_imagery/dataset/label_qa/`.

## 1. External positives (all 287 reviewed visually)

**Method:**
- **Crop check:** every positive was viewed as a 475 m crop centred on Yanliang's point (24 sheets).
- **Tile check:** the 54 that looked doubtful in the crop were re-viewed as full 1.9 km tiles.
- **Grade:** a layered or secure perimeter visible at or near the point.

| Grade | train | val | Meaning |
|---|---|---|---|
| clear | 49 | 16 | Double fence or patrol-road perimeter at the point |
| plausible | 137 | 31 | Point inside or on the edge of a secured nuclear site; fence partly visible |
| plausible at tile scale | 33 | 7 | Signature not at the point but clearly in the tile (point inside a large plant) |
| **no secure perimeter visible** | 9 | 1 | Urban, campus or business-park setting; no layered fence identifiable at this zoom |
| **point likely off-site** | 3 | 1 | Facility in the tile, but the point lies outside it (forest, road, shoreline) |

**Point likely off-site:**
- Kashiwazaki-Kariwa
- Shimane (val)
- SAFARI-1
- Heysham

**No secure perimeter visible:**
- All-Russian Sci. Research Inst. of Exp. Physics
- Fuel Handling Plant
- HTR-10
- IFA Institute (val)
- IPPE
- IR-8
- Kurchatov Institute
- Neptune (Raynesway)
- PRR pulsing reactor
- Yibin fuel fabrication

**Out-of-sample model evidence** (val positives only, mean over the 9 frozen runs): 3 of 56 val positives score below 0.5 on average:
- Plant 272 conversion (0.37)
- RWTDF (0.46)
- Maišiagala RWSF (0.48)

All three were graded plausible visually; they are small or low-contrast sites rather than mislabels. The two flagged val positives score 0.58 (IFA) and 0.93 (Shimane), so the model evidence doesn't flag them. Train positives have no out-of-sample score, and none was invented.

**Co-location:** 36 positive pairs have tile centres closer than 1.9 km (overlapping tiles).
- All 36 are within one split group, so there is no train/val leakage.
- 5 pairs are under 0.5 km and are near-duplicate views:
  - IR-8 / Kurchatov
  - HTR-10 / NHR-5
  - SNF dry storage / Wolseong
  - Hope Creek / Salem
  - Cernavoda / DICA

  These double-weight a few sites; I propose keeping them, since they are distinct facilities and down-weighting would be a design change.

**Status of the 14 flagged positives (original screening note; superseded by §1a):** all 14 stayed flagged, pending confirmation from Yanliang and Qiqi. None was removed, re-centred or down-weighted.
- **Point likely off-site (4):** Kashiwazaki-Kariwa, Shimane (val), SAFARI-1 and Heysham. If one is confirmed wrong, it would be excluded or re-centred.
- **No secure perimeter visible (10):** the underlying nuclear-site identity may well be correct. But the supervised target is the visible layered-fence / secure-facility signature, so these may still be label mismatches for this task, even if the site is right. They should be treated as possible label noise for the fence-signature target, not as confirmed-correct positives.
- **The 36 overlapping positive pairs stay**, because every pair lies within one split group.
- **Shimane** is a positive in the frozen 118-tile shared validation set. That set is not redefined, even if Shimane is confirmed wrong; a post-training sensitivity result with Shimane excluded would be reported instead.

The full list is in `positive_proposed_corrections.csv`. Its `proposed_action` column only records options pending confirmation; none has been applied.

## 1a. Final status of the 14 flagged positives (2026-10-21)

All 14 were reviewed by Yanliang and **confirmed valid**. They stay in the training and validation sets exactly as labelled:
- no coordinate, label, split or weight was changed;
- the frozen manifests, the 118-tile shared validation set and every frozen result therefore stand as they are;
- the Shimane-excluded sensitivity mentioned above is not needed.

In `positive_proposed_corrections.csv`, the 14 rows now have `status = reviewed_confirmed_valid`, `reviewer = Yanliang`, `review_date = 2026-10-21`, and `proposed_action` says to keep them as labelled. The private per-positive table (`external_imagery/dataset/label_qa/positive_qa_table.csv`) gained an `expert_review` column with the same outcome for these 14. Its other columns, including the original `visual_status` screening notes, are unchanged, to keep the provenance of why each was flagged.

The 10 "no secure perimeter visible" cases are therefore confirmed facility identities, not label noise. They remain harder examples of the target, where the layered perimeter is not obvious in the 1.9 km tile imagery used here (about 1.5 m/px effective model input).

## 2. Benchmark facility map (Saratov 24 + Natanz 1 + Fordow 1 in the denominator)

**Clean:** for every facility:
- all footprint tiles have imagery;
- every footprint tile is human-positive.

The flagged items are below; full rows are in `benchmark_facility_checks.csv` and `benchmark_facility_centroids.csv`.

1. **Fordow footprint is probably incomplete.** The z16 label centroid falls in `r15_c16`, which is not in the footprint (`r14_c15; r15_c15; r16_c16`). This adds evidence for open author question A3 (two clusters about 3.4 km apart; the manuscript's double fence east of the complex is unlabelled).
2. **Facilities whose z17 label lands in a neighbouring z16 tile:**
   - SAR-SILO-07: about 1.1 km between the z16 and z17 centroids.
   - SAR-SILO-15
   - SAR-SILO-20
   - SAR-SILO-21: about 1.05 km between centroids.
   - SAR-WHS-01

   In each case the footprint may need that neighbour tile. SAR-SILO-04 (labelled "under renovation", 1.39 km z16/z17 disagreement) has a z17 centroid that falls in no z16 parent tile.
3. **Centroids outside every parent tile:** SAR-SILO-20 and SAR-WHS-02 (z16). Their footprints are multi-tile, so this is probably a projection artefact of the centroid check, not a label error. Verify with Qiqi.
4. **Tile label vs facility map.** 106 of the 146 human-positive benchmark tiles lie outside every facility footprint. They are non-nuclear layered-fence sites (oil/gas, military bases, airbases, prisons, power, industry). So `human_target` measures "layered-fence signature", not "nuclear facility". This explains why tile AUROC and facility recall can rank conditions differently, and the paper should say so explicitly.
5. **Open questions.** Open author questions A1–A6 in `facility_mapping_questions.md` are unchanged. No correction is proposed without author confirmation.

**Impact on the frozen A/B/C results:** none. The issues above touch 6 Saratov facilities and the Fordow footprint, and could change those facilities' first-hit areas. They should be re-scored once Yanliang and Qiqi confirm; that needs no retraining.
