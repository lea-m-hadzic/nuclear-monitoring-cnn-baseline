# Full-resolution imagery QA notes (2026-10-08)

Manual visual QA of a small stratified sample of the 2560×2560 tiles shared by Qiqi. **QA only: no label,
facility mapping, split or benchmark semantics were changed.** Coordinates are in native pixels (0.742 m/px).

## Cache-level findings

1. **Natanz: RESOLVED (2026-10-09).** History: the first shared Natanz folder resolved only 246 of 776 parents; its
   other 261 PNGs came from a grid about 1° further south (lat 32.46–32.98 vs 33.46–33.98). The missing tiles
   included the facility tile `r15_c16` and 3 of the 8 human-positive tiles. Qiqi's canonical re-upload
   (`NEW_natanz/`, `natanz_30km_tiles_auto_mosaic_reupload`) contains exactly the 776 benchmark tiles:
   - no off-grid tiles and no duplicates;
   - all 8 human-positive tiles and the facility tiles `r15_c16`, `r15_c17` and `r16_c16`;
   - the 246 overlapping files are byte-identical to the old copies (size + CRC32; SHA-256 for cached tiles).

   It is now the exclusive Natanz source, and the old folder is no longer used.
2. **Saratov z16 zip part 006 was not extracted** (213 tiles). The resolver reads those directly from the zip.
3. Fordow part 4 holds a byte-identical duplicate `…r005_c014…tile1.90km(1).png` (ignored). Saratov's scan folder
   also contains 13 tiles from another grid (ignored). `saratov_40km_tiles/` is an older 2.0 km / 673 px scan
   that the benchmark does not use.
4. All 2,940 resolved images (all three AOIs) fully decode as 2560×2560 RGB; 0 corrupt (re-checked 2026-10-09).
5. **The shared `manifest.csv` files embed a Google Maps API key** in the `url` column. Never commit `data/`.

## Imaging differences across AOIs (possible CNN shortcuts)

- **Source zoom differs:** Fordow/Natanz tiles are 4×4 mosaics of **zoom-17** imagery (native 0.488 m/px)
  resampled to 0.742 m/px; Saratov z16 tiles come from **zoom-16** imagery (0.737 m/px), i.e. different sharpness.
- **Watermark grid differs:** each source image carries "Google" plus an "Imagery ©… Airbus, Maxar…" strip, so
  Fordow/Natanz tiles show a 4×4 watermark grid and Saratov a 2×2 grid, plus visible mosaic seams/colour steps.
  Within one AOI these are constant (harmless for spatial CV); across AOIs (LOAO) they encode the region.
  Consider masking or cropping watermark strips, and report LOAO with this caveat.

## Resolution: is double fencing visible?

| Example | native (0.74 m/px) | 1280 px (1.48 m/px) | 640 px (2.97 m/px) |
|---|---|---|---|
| Saratov `r13_c3` silo perimeter (two concentric fences ~15–20 m apart, each with a track) | crisp | both fences clearly separate | fences visible but blurred into bands |
| Saratov `r2_c17` silo enclosure (3 concentric lines) | crisp | clear | visible, lines start to merge |
| Saratov `r42_c22` warhead-storage berms/perimeter | clear | clear | still visible (large structure) |
| Fordo `r16_c16` hall perimeter (double line around hall) | visible | visible | merges into one dark band |
| Natanz `r16_c16` inner compound fences | visible | mostly visible | largely lost |

**Recommendation: 1280 px** (2× box reduce). It keeps fence pairs ≥ ~10 m apart separable at about ¼ of native
compute. 640 px loses the closely spaced pairs (Fordo, Natanz); native 2560 px costs ~4× more (1.28 s/img to train
on the M5 Max) for little visible gain.

## Label plausibility (sampled; no relabelling)

| Stratum | Tiles inspected | Observation |
|---|---|---|
| TP | Saratov `r42_c22` (WHS), `r13_c3` (silo), `r15_c7` NE (silo); Fordow `r14_c15`, `r16_c16` NW; Natanz `r16_c16` | Plausible: layered perimeters visible. |
| FN | Saratov `r41_c23` (WHS, −8), `r5_c10` (silo under renovation), `r2_c17` (silo, −7); Fordow `r9_c3` (military base, −8) | `r41_c23`, `r5_c10`: facility only partly inside the tile (edge), so the label relies partly on the neighbouring tile. **`r2_c17`: unmistakable silo enclosure with three concentric fence lines.** **Fordow `r9_c3`: only single linear tracks visible in this tile, even at native resolution; the label likely relies on a large base perimeter outside the tile.** |
| FP | Saratov `r11_c28` (FP - single, airfield) | Plausible: a single fence line. |
| TN | Saratov `r8_c29`, Natanz `r10_c23`, Fordow `r22_c27` | Plausible: no compounds; a single-walled farm plot in Fordow. |
| Flagged | Natanz `r8_c15` (parent TN vs quadrants TP); Fordow `r9_c2` (zoom-out only, parent FN) | `r8_c15`: walled power/industrial compound visible, which supports the quadrant-union target. `r9_c2`: a fence line with a parallel patrol track: a borderline "layered" judgement. |
| Inconclusive | Fordow `r10_c3` (TP - military base, 7) | The inspected crop did not show the base; needs a whole-tile review. |

## 21st silo (`SAR-SILO-01`, z16 `r2_c17`): new evidence for the authors

The enclosure is clearly a silo with layered fencing. It sits at the tile's top edge, approximately
**40.0 km** from the Saratov AOI centre (51.52 N, 46.00 E), i.e. on the 40 km boundary. The z16 tile (centre at
39.2 km) covers it; the z17 scan does not (nearest z17 tile centre 0.8 km away; z17 centres stop at 40.04 km).
Likely explanation: the manuscript's "20 silos within 40 km" excludes this boundary silo. Still needs author
confirmation; the default (excluded from the denominator) is unchanged.
