# Augmentation Study — Run Record

Rewritten 2026-09-27, replacing the 2026-09-14 draft (which covered only the
none/adaptive pair at one seed). Records the full augmentation study: what was
run, what came out, and which of the paper's claims survive. Every number here
was recomputed from the committed output files on `benchmark_branch`.

> Where a statement is an interpretation rather than a measurement, it says
> so. §10 lists what to re-check independently.

---

## 0. Read this first

1. **Augmentation helps every architecture.** The sample-adaptive policy beat
   no augmentation on 8 of 8 baselines (+1.45 accuracy points, Wilcoxon
   p=0.0078, the smallest value the test can return at n=8) and on the
   searched architecture (+3.09 points).
2. **Per-sample adaptivity is not what produces the gain.** The
   constant-strength control — same operators and probabilities, one fixed
   strength per operator, no loss ranking — captures nearly all of it.
   Adaptive minus constant is +0.40 points on the baselines (p=0.25) and
   +0.63 on z* (McNemar significant in 1 of 3 seeds). Widening the operator
   bounds did not change this (§5.4).
3. **The searched policy beats the legacy fixed policy only on z\***, where
   the gap is large (+4.36 points, +20.7 net test images). On the 8 pretrained
   baselines the two are indistinguishable (+0.18 points, p=0.48).
4. **The legacy fixed policy makes z\* worse than no augmentation at all**
   (0.8572 vs 0.8699).
5. **The searched model no longer beats every baseline.** Against
   DenseNet-121, both under the adaptive policy, both averaged over 3 seeds:
   0.9008 vs 0.9093 accuracy, and DenseNet leads on macro-F1 and macro-AUC
   too. McNemar cannot separate them (p=0.72, 0.87, 0.12).
6. **Single-seed results misled us twice** (§7). Everything here is 3 seeds.

---

## 1. What was compared

Four augmentation arms, applied to 8 external baselines and to the searched
architecture z*, on the same patient-disjoint split, 3 seeds each:

| Arm | What it is |
|---|---|
| **none** | resize + grayscale + normalize only |
| **fixed** | the legacy 2-operator policy (`outputs/fixedda_run/augmentation_run/selected_legacy_policy.json`): brightness + rotation, p=1, magnitude sampled up to each operator's bound, no LossCache |
| **constant** | h*'s operators, order and probabilities, each held at its rank-averaged strength for every image, no LossCache |
| **adaptive** | h* itself (`outputs/main1/augmentation_run/selected_policy.json`): per-image strength from that image's current loss rank |

`constant` exists to separate two things the fixed-vs-adaptive comparison
confounds: the richer operator set and the per-sample adaptivity.

---

## 2. Code (all committed and pushed on `benchmark_branch`)

| Commit | What |
|---|---|
| `ab8375a` | `run_baseline.py --augmentation-policy`: baselines can train under the sample-adaptive policy |
| `d3e45fe` | cherry-pick of `6441973`: the legacy fixed 2-op code |
| `50a5b3c` | `run_baseline.py --fixed-augmentation-policy` |
| `b815a85` | per-sample `test_predictions.csv` for baselines, plus `scripts/backfill_predictions.py` |
| `c7c3b39` | `scripts/check_convergence.py` and the pre-registered convergence rule |
| `98e8824` | per-sample predictions in final training |
| `765c9de` | `run_final_training.py --seed` and `--legacy-augmentation-output-dir` |
| `d6f7b0a` | constant-strength control (`--constant-strength-policy`, `--constant-strength-output-dir`) |
| `080638e` | exhaustive search over all 21 operator pairs |
| `877aee3` | per-run `--magnitude-bounds`, recorded in each run's `magnitude_bounds.json` |

Test suite: 205 tests pass.

**Uncommitted on the Mac:** the corrected `optim.py` docstring and
`run_baseline.py` optimizer help text (§8.3), and the `visualization.py`
figure settings.

---

## 3. Runs

All on one GCP VM (`g2-standard-8`, one L4), conda env `moenas`,
torch 2.3.1+cu121, torchvision 0.18.1+cu121, **timm 1.0.29**, same split
throughout (`split_indices.json` sha256 prefix `4ca8de66ae81`,
2102 train / 488 val / 474 test).

| Group | Runs |
|---|---|
| Baselines: 8 models × 4 arms × 3 seeds | 96 |
| z*: 4 arms × 3 seeds | 12 |
| z*: 2 arms × 2 extra bound levels × 3 seeds | 12 |
| Exhaustive pair search: 21 pairs × 3 repeats | 63 trials |

EfficientNetV2-M and DeiT-Small run at **100 epochs**, every other baseline at
50, because the pre-registered convergence rule flagged them (§6). z* runs at
200 epochs from scratch.

The original `baseline_<model>_96` directories (old VM, unknown timm version)
were superseded by `_none` re-runs on this VM. They reproduced almost exactly:
7 of 8 identical to 4 decimals, VGG-16 different by 1 test image. They are
kept as a cross-machine reproducibility check.

---

## 4. Integrity checks (all passed)

- **127 prediction files**: 474 rows each, identical image order across every
  run, and each file's accuracy equals its own `test_metrics.json`.
- **Every run**: split hash matches `main1`, recorded seed matches the folder,
  recorded arm matches the flag used, epoch budget as intended.
- **Policies**: each augmented run's copied policy file is byte-identical to
  the source in `main1` / `fixedda_run`.
- **Constant arm**: all 27 runs record 7 per-operator strengths matching
  `rank_averaged_strengths` for 2102 training samples.

---

## 5. Results

### 5.1 Baselines, 3-seed mean accuracy

| Model | none | fixed | constant | adaptive |
|---|---|---|---|---|
| ResNet-18 | 0.8622 | **0.8882** | 0.8854 | 0.8790 |
| ResNet-34 | 0.8734 | 0.8910 | 0.8924 | **0.8931** |
| VGG-16 | 0.8959 | 0.8861 | **0.9001** | 0.8980 |
| DenseNet-121 | 0.8966 | **0.9107** | 0.8952 | 0.9093 |
| EfficientNetV2-S | 0.8678 | 0.8854 | 0.8819 | **0.8952** |
| SE-ResNet-50 | 0.8748 | **0.8840** | 0.8706 | 0.8805 |
| EfficientNetV2-M | 0.8868 | 0.8903 | **0.8952** | 0.8910 |
| DeiT-Small | 0.8622 | 0.8861 | 0.8826 | **0.8896** |
| **Mean** | 0.8775 | 0.8902 | 0.8879 | **0.8920** |

Per-model mean ± SD: `paper/tables/accuracy_by_arm.tex`.

### 5.2 Pre-registered test (Wilcoxon signed-rank across the 8 models, seed-averaged accuracy)

| Comparison | Difference | Models won | p |
|---|---|---|---|
| adaptive vs none | +1.45 pts | 8/8 | **0.0078** |
| fixed vs none | +1.27 pts | 7/8 | **0.039** |
| constant vs none | +1.05 pts | 6/8 | **0.039** |
| adaptive vs constant | +0.40 pts | 5/8 | 0.25 |
| adaptive vs fixed | +0.18 pts | 5/8 | 0.48 |

### 5.3 Searched architecture z*, 3 seeds

| Arm | Seed 1 | Seed 2 | Seed 3 | Mean ± SD |
|---|---|---|---|---|
| none | 0.8861 | 0.8460 | 0.8776 | 0.8699 ± 0.0211 |
| fixed | 0.8418 | 0.8692 | 0.8608 | 0.8572 ± 0.0140 |
| constant | 0.8882 | 0.8903 | 0.9051 | 0.8945 ± 0.0092 |
| adaptive | 0.9114 | 0.8987 | 0.8924 | **0.9008 ± 0.0097** |

- adaptive − none **+3.09**, of which constant − none accounts for **+2.46**
- fixed − none **−1.27**: the legacy policy hurts z*
- adaptive − fixed **+4.36** (McNemar +20.7 images, significant in 2 of 3 seeds)

The paper's original 6.96-point adaptive-vs-fixed gap shrinks to 4.36 once
both sides are averaged over seeds: `main1` (0.9114) and `fixedda_run`
(0.8418) were each the luckiest seed on their side.

### 5.4 Does adaptivity need wider bounds? (pre-registered follow-up)

Hypothesis: h*'s operators are bounded so tightly (≤15° rotation, ≤0.10
translation, crop ≥0.85) that adaptivity has little room to express itself.
Bounds were widened using tumour masks from the **training** cases only
(`cjdata.tumorMask`, 2590 cases, no test data), keeping the full tumour inside
the frame.

| Bounds | rot. / trans. / crop | Adaptive | Constant | adaptive − constant |
|---|---|---|---|---|
| L1 current | 15° / 0.10 / 0.85 | 0.9008 ± 0.0097 | 0.8945 ± 0.0092 | +0.63 pts (+3.0 imgs) |
| L2 mid | 30° / 0.13 / 0.75 | 0.8797 ± 0.0146 | 0.8924 ± 0.0076 | **−1.27 pts** (−6.0 imgs) |
| L3 wide | 45° / 0.16 / 0.65 | 0.8882 ± 0.0092 | 0.8812 ± 0.0199 | +0.70 pts (+3.3 imgs) |

**The prediction failed.** The gap does not grow with the bounds and its sign
flips. Both arms also got *worse* as bounds widened, so the original MRI-safe
bounds were already at or past the useful strength for this model.

Supporting geometry (training cases only): the furthest tumour pixel sits at
0.416 of the canvas half-width, inside the inscribed circle in 100% of images,
so rotation never pushes tumour out of frame; 95% of images tolerate
translation up to 0.170; crop scale 0.65 retains the whole tumour in 99.8%;
a 5% erase patch is larger than the entire tumour in 96.4% of images, which is
why erasing was **not** widened.

### 5.5 Exhaustive search over the legacy policy space

All 21 operator pairs, 3 repeats each, same trial protocol and same shared
initial weights as the legacy GA (a test asserts the weight equality).

- **Winner: rotation+contrast**, 0.8340 ± 0.0157 validation macro-AUC.
- The legacy GA's pick, **brightness+rotation, ranks 8th** at 0.8138 ± 0.0162.
- Gap: +0.0202, about 1.56 standard errors — not a clean separation.
- All 21 pair means span 0.7629 to 0.8340.
- The legacy GA explored only 10 of 21 pairs and spent 8 of its 25 trials on
  one pair, whose scores ranged 0.7917–0.8555. A single 10-epoch trial has a
  standard deviation of about 0.020, as large as most gaps between pairs.

**Not yet done:** under the pre-registered rule the fixed arm should be re-run
with rotation+contrast for every model and seed (~10 h). Until then the fixed
arm is the legacy GA's own choice.

*Interpretation, not measurement:* both searches landed on rotation paired
with an operator the preprocessing largely cancels (§8.1), i.e. effectively
rotation alone.

### 5.6 Paired agreement (McNemar)

`paper/figures/mcnemar_matrix.{png,pdf}` and
`paper/tables/mcnemar_by_arm.tex`. Exact McNemar on paired per-image
correctness, run separately per seed. Highlights:

- adaptive − none is positive in **9 of 9** models.
- adaptive − fixed scatters around zero on the baselines (−4.3 to +5.7) but is
  **+20.7 for z\***.
- fixed − none is **−6.0 for z\***, positive for most baselines.

### 5.7 z* versus the best baseline

Both under the adaptive policy, 3 seeds each:

| Metric | z* (0.89M params) | DenseNet-121 (6.96M) | z* higher in |
|---|---|---|---|
| Accuracy | 0.9008 | **0.9093** | 1 of 3 seeds |
| Macro-F1 | 0.8871 | **0.9008** | 0 of 3 |
| Macro-AUC | 0.9728 | **0.9827** | 0 of 3 |

McNemar per seed: p = 0.72, 0.87, 0.12. Not separable on 474 images, but
DenseNet-121 leads on every metric on average.

---

## 6. The convergence rule

`src/brainmri_nas/training/convergence.py`, thresholds fixed before the
results they were applied to: flag a run if mean validation accuracy rises
more than 0.5 points, or macro-AUC more than 0.2 points, over the final 20% of
epochs versus the 20% before it, averaged over seeds within an arm.

- At 50 epochs it flagged **DeiT-Small** and **EfficientNetV2-M**.
- Both were retrained at 100 epochs **in all arms**, so the longer schedule is
  never confounded with augmentation.
- At 100 epochs neither is flagged.
- `flag_models` refuses to treat two runs of the same seed as two seeds; the
  default glob excludes the superseded originals.

Disclose: a 100-epoch run refreshes loss-cache ranks every 2 epochs rather
than every 1 — the same rule applied to a different run length.

---

## 7. Two single-seed results that did not survive

1. **EfficientNetV2-M at 100 epochs, adaptive**: 0.9241 at seed 1, but 0.8819
   and 0.8671 at seeds 2 and 3 (mean 0.8910, against fixed's 0.8903). The
   "+5.5 point jump" was luck.
2. **z\* adaptive vs fixed**: 6.96 points on single seeds, 4.36 averaged.

Typical seed-to-seed SD of baseline accuracy is 0.0090, about 4 test images.

**Same seed reproduces exactly.** Re-running z*+adaptive at seed 1 on the new
VM matched `main1` at every epoch checked (1, 2, 50, 170, 200) and on the
final test metrics. The earlier claim in `HANDOFF.md` that runs are "not
bit-reproducible" was an inference from settings, not a measurement.

---

## 8. Findings about the augmentation operators themselves

### 8.1 Two of the seven operators do almost nothing
`PerImageNormalize` standardizes each image by its own mean and standard
deviation, which undoes a brightness change and largely undoes a contrast
change. Measured on 200 real training images, as mean relative change to the
network's input:

| Transform at its maximum | Change |
|---|---|
| brightness ×0.8 / ×1.2 | 1.2% / 1.0% |
| contrast ×0.8 / ×1.2 | 1.2% / 4.0% |
| rotation 7.5° / 15° | 35% / 48% |

Consequences: the legacy fixed policy is effectively rotation-only; h*'s
effective operators are translation, crop, flip and occasional rotation and
erasing; and both policy searches spent choices on operators that cannot act.

### 8.2 Horizontal flip has no strength axis
`MAGNITUDE_RANGES["horizontal_flip"] = (0, 0)`, so flip is controlled by
probability alone and is never adaptive.

### 8.3 The DeiT optimizer note was wrong
The claim that SGD "plateaued" for DeiT was a misreading: 0.2916 is the
label-smoothing floor (0.2911 for ε=0.1, K=3), i.e. the model had fit the
training set. The DeiT paper reports AdamW and SGD performing similarly for
fine-tuning (Touvron et al. 2021, §6, Table 8: 83.1 vs 83.1); SGD only hurts
pre-training from scratch. Caveat: "fine-tuning" there means fine-tuning at
384px on ImageNet, not transfer to another domain, so this supports dropping
the AdamW rows but should not be overstated. The corrected docstring is
written but not yet committed.

---

## 9. What the paper must change

1. **"The searched model outperforms every external baseline"** — no longer
   true (§5.7). It is second of nine, within noise of the best, with 7.8×
   fewer parameters.
2. **All results become mean ± SD over 3 seeds**, in tables and prose.
3. **The augmentation claim becomes:** augmentation improves every
   architecture tested; the searched policy beats the legacy fixed policy on
   the searched architecture; per-sample adaptivity is **not** demonstrated to
   be the active ingredient (§5.2, §5.3, §5.4).
4. **Table 6's "same protocol" caption** is now true — all arms share the
   split, protocol, machine and software.
5. **§6.2's "isolates the contribution of sample-adaptivity"** must go; the
   constant-strength control is what isolates it, and it came out null.
6. **TOPSIS weights**: the text now matches the runs at (0.35, 0.35, 0.30);
   `configs/figshare.yaml` still says (0.375, 0.375, 0.25).
7. **Report the operator measurement** in §8.1 — two of seven operators are
   inert under this preprocessing.
8. **Disclose**: the convergence rule and the 100-epoch reruns, the bounds
   sensitivity result, and the exhaustive search ranking.

---

## 10. What to check independently

1. Spot-check any run: `outputs/<run>/test_metrics.json` against the tables.
2. `python scripts/check_convergence.py` — expect DeiT-Small and
   EfficientNetV2-M flagged at 50 epochs, nothing at 100.
3. `python scripts/backfill_predictions.py --runs "outputs/baseline_*_96_*"` —
   expect every run to skip (CSV present) and no mismatches.
4. The exhaustive ranking in
   `outputs/exhaustive_pair_search/exhaustive_pair_summary.json`.
5. The bounds used by any run: `outputs/<run>/magnitude_bounds.json`.
6. The claims in §5.4 and §5.5 before writing them up — both rest on 3 seeds
   or 3 repeats, and §5.5's top two pairs are 1.56 standard errors apart.
