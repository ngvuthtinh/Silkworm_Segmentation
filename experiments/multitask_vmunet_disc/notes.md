# 📝 multitask_vmunet_disc — baseline + per-pixel embedding to re-join pieces of one larva

## Goal
Limitation L1: a larva hidden by a leaf into several disjoint pieces is **one id** in the labels, but `split_instances` returns **one region per connected piece**, so the pieces are counted as separate larvae (ceiling `inst_f1` ≈ 0.794 on composed images, see `multitask_vmunet/notes.md`). This experiment adds a way to say "these two pieces are the same larva" that does **not** depend on the pieces touching.

## What is different from the baseline (`multitask_vmunet`)
| | Baseline | This experiment |
|---|---|---|
| Output channels | 3 (body, boundary, disease) | 3 + `n_embed` (default 8) embedding channels |
| Loss | `L_body + L_bnd + L_dis` | same + `λ_embed · DiscriminativeLoss` |
| Post-processing | `split_instances` | `split_instances`, then `merge_fragments` (join regions whose mean embedding is closer than `merge_thr`) |
| Everything else (data, lr, epochs, batch, seed, λ_body/bnd/dis, thresholds) | — | **identical**: `DiscConfig` inherits `MultiTaskConfig` and only adds fields |

> ⚠️ **One-change rule.** The project rule is "loss *or* architecture, never both". Here the new output channels (architecture) and the discriminative loss are inseparable (the loss has nothing to train without the channels), so this is a **deliberate exception**: record it as one experiment and say so in the `Đổi gì` cell of `Reports/experiment_results.xlsx`. Ask the advisor if she wants it split.

## Method (De Brabandere et al., 2017, arXiv:1708.02551)
- Each pixel inside a larva is mapped to a point in `n_embed`-dim space. `L = L_var + L_dist + 0.001·L_reg`:
  - `L_var` pulls pixels of the same larva towards their cluster centre (hinge `delta_var`), averaged **per larva** then over larvae.
  - `L_dist` pushes centres of different larvae apart (hinge `2·delta_dist`). Needs `delta_dist > 2·delta_var`.
- **Why it can fix L1**: the loss groups pixels by *id*, not by connectivity, so the two visible pieces of a hidden larva share one cluster. (The paper itself says watershed-style methods "cannot handle occlusions where instances are separated into multiple pieces", §2.)
- **Why the merge threshold 1.5**: with `delta_var = 0.5`, `delta_dist = 1.5`, two pieces of one larva have mean embeddings ≲ 1.0 apart, and pieces of different larvae ≳ 2.0 apart. Any threshold in (1.0, 2.0) separates them in the ideal case; 1.5 is the middle. **Tune it on `valid` only** (`--merge-thr`), never on `test`.
- `split_instances` is kept: the boundary channel still separates larvae that *touch*; the embedding only merges pieces that are *apart*.
- Implementation is a vectorised re-write from the paper's equations (`src/losses.py::DiscriminativeLoss`), checked against a naive per-cluster reference in `test_embedding.py`. It is **not** a copy of `Wizaron/instance-segmentation-pytorch` (Python 2 / old PyTorch, GPLv3); differences: per-larva averaging of `L_var` as in paper eq. (1), no one-hot tensors, no count head, no KMeans.

## Run
```bash
python -m experiments.multitask_vmunet_disc.test_embedding                    # unit checks, no GPU
python -m experiments.multitask_vmunet_disc.train --smoke-test --gpu 1        # 5 train + 3 val batches
python -m experiments.multitask_vmunet_disc.train --gpu 1 --out-dir Reports/Meeting_2026-10-30/checkpoints/multitask_vmunet_disc
python -m experiments.multitask_vmunet_disc.evaluate --checkpoint <run>/checkpoints/best.pth --split valid --merge-thr 1.2
python -m experiments.multitask_vmunet_disc.evaluate --checkpoint <run>/checkpoints/best.pth --split test
python -m experiments.multitask_vmunet_disc.evaluate --checkpoint <run>/checkpoints/best.pth --split test_real
```
Run in `tmux`; `--resume <run dir>` works (the embedding size and δ are restored from the run's `config.yaml`). Evaluation reads `n_embed`, `delta_*` and `merge_thr` from the run's `config.yaml` automatically.

## Reading `train_log.csv` (new columns)
| Column | Meaning |
|---|---|
| `loss_embed`, `loss_embed_var`, `loss_embed_dist` | discriminative loss and its two main terms. `var` high = pixels of one larva are not grouped; `dist` high = different larvae overlap in embedding space |
| `inst_f1`, `e2e_f1` | **with** merging (main score, used to pick `best.pth`) |
| `inst_f1_nomerge`, `e2e_f1_nomerge` | the same predictions **without** merging = what the baseline post-processing would give. **Their difference is the effect of the merge step** |
| `n_regions_before_merge`, `n_regions_after_merge` | total regions over the validation set before/after merging |

## What to look at first
1. `inst_f1` vs `inst_f1_nomerge` on `test` (composed): does it go above the old ceiling 0.794? If not, either the embeddings do not group the pieces (`loss_embed_var` stays high) or `merge_thr` is off (tune on `valid`).
2. `test_real` must not get worse (real photos have one blob per larva by construction, so merging should change little there; if it drops, the merge is joining different larvae).
3. `grasserie_recall` must not drop (missing a sick larva is the costly error).

## Risks / caveats
- **Generalisation (L2) may get worse.** The paper (§3.3) warns that this holistic approach does well when images look alike and badly when object layouts vary, and gives "trained on single-object images, fails on images with many objects" as the example. That is close to L2 here. Check the 2 out-of-domain photos in `data/test` too.
- **Shortcut risk on composed data**: leaves are synthetic polygons pasted over larvae (`utils/augment_mixed_10k.py`). The network may learn "cut by a polygon" instead of "same larva". `test_real` cannot show L1 (every blob is one larva there), so the real-photo effect of this experiment is hard to measure until a small hand-labelled set of real multi-larva photos exists.
- Merging uses single linkage over region mean embeddings; a chain A~B~C joins all three even if A and C are far apart. Not a problem in the ideal geometry above, but watch for it if two larvae are merged on `valid`.
- Baseline B1 (`2026-10-14_23-36-58`) was trained on the 20,000-image version before the 11-image fix (see `DATASET_STRUCTURE.md` §2.5); this experiment uses the default `data/sam3_aug20k_v3` (19,989 images). The difference is tiny; retrain the baseline on the same data for a strictly clean comparison.
- Pretrained VMamba weights are not present (`models/vmunet/pre_trained_weights/vmamba_small_e238_ema.pth`), same as the baseline run: training from scratch.
- Ignore regions (`valid = 0`) are excluded from the embedding loss like from every other term.

## Results table
After training + evaluation, save scores as `eval_*.txt` under `results/<run>/eval_data_<x>/`, run `python utils/export_results_xlsx.py`, then fill the three manual columns (`Mã`, `Đổi gì`, `Ghi chú`) yourself. Not done yet: no full training has been run.
