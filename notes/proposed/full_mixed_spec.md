# FULL_MIXED: final supervised CNN baseline (specification; NOT launched)

**Status:** this is a specification only. Nothing has been implemented or run. FULL_MIXED waits for Lea's approval, which comes after Yanliang and Qiqi confirm the 14 flagged positives. The rule below is written into `freeze_full_mixed.json` before any run, and is not changed afterwards.

## Data

- **Positives:** all QA-approved external train positives, decided after label confirmation. Until then, the 14 flagged positives stay flagged and none is removed.
- **Hard negatives:** all 126 confirmed hard train negatives.
- **Nearby pool:** the 638 clean nearby train negatives (`nearby_negative_selected` and not auto-rejected).
- **Checkpoint selection:** the frozen 118-tile shared validation set (checksum `ade28778…`), unchanged. If Shimane is later confirmed erroneous, the early-stopping set is NOT redefined. Instead, a post-training sensitivity result with Shimane excluded is reported.
- **Excluded:** benchmark AOIs never contribute any data.

## Per-epoch sampling rule (1:1 nearby:hard)

Each epoch's training set is: all positives, all 126 hard negatives, and 126 nearby negatives.

The nearby negatives are chosen as follows:

1. **Shuffle once.** Order the nearby pool by `SHA-256("external-v1" | "full_mixed_nearby" | seed | tile_id)`, giving a fixed permutation per seed.
2. **Draw in blocks.** Epoch e takes the next 126 tiles from the permutation, continuing from where epoch e−1 stopped.
3. **Wrap.** When the permutation runs out, continue from the start of the same permutation. The whole pool is used before any tile repeats: one cycle is 638 / 126 ≈ 5.06 epochs.
4. **Log.** The selected tile IDs for each epoch are written to `epoch_nearby_selection.csv` (epoch, position, tile_id), and its checksum is recorded in `run_config.json`.

The batch order within an epoch uses the existing seeded DataLoader shuffle.

## Everything else is identical to the frozen ablation

- **Model:** ResNet-50 with ImageNet V2 weights, single logit, 1280 px input.
- **Loss:** weighted BCE with `pos_weight = n_neg / n_pos` of the epoch training set, which is constant across epochs.
- **Training:** AdamW (lr 1e-4, wd 1e-4), cosine schedule over 20 epochs, batch size 4, rot90 and flip augmentation, fp32.
- **Masking:** the geometric watermark mask.
- **Early stopping:** on shared-val PR-AUC, minimum 5 epochs, patience 5.
- **Seeds:** 0, 1 and 2.

## Disclosure for the paper

The mixed design was chosen from the A/B/C ablation, which was evaluated on the benchmark AOIs. The shared-validation result for C agrees with that choice, but it is distributionally favourable to C, because the validation negatives are themselves a nearby/hard mixture. No hyperparameter, threshold or epoch count was tuned on benchmark results.
