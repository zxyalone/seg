# 4 Experiments and Results

This section first fixes the data, implementation, and evaluation protocol (Sections 4.1–4.4). All of these choices were frozen on the validation cases before any test case was evaluated. Results follow in Sections 4.5–4.10. Placeholders of the form `[[TBD]]` mark values that are not yet available.

## 4.1 Datasets

**LiTS (development and internal test).** We use the public training set of the Liver Tumor Segmentation Benchmark (LiTS) \cite{bilic2023lits}. It contains 131 contrast-enhanced abdominal computed tomography (CT) volumes contributed by seven clinical sites. The cohort mixes primary liver tumors and metastases, pre- and post-therapy scans, and hypo- and hyper-dense lesions \cite{bilic2023lits}. Across the full benchmark, in-plane resolution ranges from 0.56 to 1.0 mm and slice thickness from 0.45 to 6.0 mm. In the training set, the median in-plane resolution is 0.76 mm and the median slice thickness is 1.0 mm \cite{bilic2023lits}. Each voxel is labelled background (0), liver (1), or tumor (2). The tumor label covers any neoplastic lesion, regardless of origin. The LiTS test labels are not public. LiTS is also the liver task (Task03) of the Medical Segmentation Decathlon (MSD) \cite{simpson2019msd}, so MSD Task03 cannot serve as an independent test set either.

**Data split.** We split the 131 public cases at patient level into 83 training, 21 validation, and 27 test cases. This is fold 0 of the nnU-Net dataset `Dataset102_LiTS`. The three sets do not overlap. Three test cases (LiTS_032, LiTS_106, LiTS_115) contain no tumor voxels. The remaining 24 test cases contain tumor. The training cases contain 518 tumor lesions (3D connected components, [[TBD: confirm minimum-size criterion used when counting, e.g., ≥ 50 voxels]]). The validation cases contain 91 lesions of at least 50 voxels. We do not count test-set lesions until the final evaluation.

**Preprocessing.** All volumes are resampled to 1.0 × 0.78 × 0.78 mm (z, y, x), the target spacing chosen by the nnU-Net planner. Intensities use the nnU-Net CT normalization: clipping to the 0.5–99.5 percentiles of foreground intensities, then z-scoring with foreground statistics \cite{isensee2021nnunet}. This global affine mapping preserves the contrast-to-noise ratio (CNR) defined in Section 4.3.

**External test set.** An independent external cohort is used for testing only: [[TBD: TCIA Colorectal-Liver-Metastases or HCC-TACE-Seg]]. No model or threshold is tuned on it.

**Why 3D-IRCADb is not used as an external test.** Some prior studies train on LiTS and test on 3D-IRCADb as a cross-dataset evaluation \cite{cheng2024egunetr}. However, 20 LiTS training scans were contributed by IRCAD, the institution that also released 3D-IRCADb, and none of the LiTS test scans came from IRCAD \cite{bilic2023lits}. 3D-IRCADb-01 contains CT scans of 20 patients \cite{cheng2024egunetr}. Patient overlap between the two collections is therefore likely and cannot be excluded. A model trained on LiTS may already have seen these patients, so 3D-IRCADb does not test generalization to unseen patients or centers. We therefore exclude it.

**Table 1.** Summary of datasets. "Resampling" is the voxel spacing (z, y, x) after preprocessing. "Tr/Va/Ts" is the patient-level split into training, validation, and test cases.

| Dataset | Role | Tr/Va/Ts (cases) | Tumor-bearing cases (Tr/Va/Ts) | Lesions (Tr/Va/Ts) | Resampling (mm) | Labels | Availability |
|---|---|---|---|---|---|---|---|
| LiTS training set \cite{bilic2023lits} (`Dataset102_LiTS`, fold 0) | Development + internal test | 83 / 21 / 27 | [[TBD]] / [[TBD]] / 24 | 518 / 91 (≥ 50 voxels) / [[TBD: counted at final evaluation]] | (1.0, 0.78, 0.78) | Liver, tumor | Public |
| [[TBD: external cohort]] | External test only | – / – / [[TBD]] | – / – / [[TBD]] | – / – / [[TBD]] | (1.0, 0.78, 0.78) after resampling | [[TBD: tumor only, or liver + tumor]] | Public (TCIA) |

## 4.2 Implementation details

PhASER-3D is implemented in the nnU-Net v2 framework \cite{isensee2021nnunet} ([[TBD: nnU-Net version / commit hash]]). All models run on a single NVIDIA RTX 3090 (24 GB) with PyTorch 2.1 and CUDA 11.8 (Python 3.10).

**Backbone.** The backbone is the nnU-Net 3D `PlainConvUNet` with six stages (32, 64, 128, 256, 320, 320 channels) and deep supervision. The first stage does not downsample, so encoder features have resolutions 128³, 64³, 32³, 16³, 8³, and 4³ for a 128³ patch. The encoder is initialized from VoComni pretrained weights \cite{wu2024voco,wu2024largescale}. Of the 292 parameter tensors, 282 are loaded; the 10 segmentation-head tensors are re-initialized because VoComni predicts 21 classes and our task has 3.

**PhASER modules.** One PFCR module is inserted on each of the five non-bottleneck encoder skips (128³ to 8³). Phase congruency (PC) uses S = 3 learnable log-Gabor scales, initial center wavelengths 3.0, 6.3 and 13.2 voxels, and bandwidth σ_onf = 0.55. The noise multiplier is initialized at k = 2.0. The fast Fourier transforms run in fp32 under automatic mixed precision (AMP). PC resolution at the 128³ skip: [[V2-pending: full resolution, or capped at 2²⁰ voxels]]. Noise threshold form: [[V2-pending: hard (Kovesi) or soft]]. The router's last layer and the residual scale γ are zero-initialized. Each module is therefore an exact identity at initialization.

**Training.** Table 2 lists all settings. The same settings are used for every ablation group (Section 4.6) unless the group definition changes them. Final-epoch weights are reported for every model; there is no checkpoint selection.

**Table 2.** Implementation and configuration of PhASER-3D on LiTS. Items marked `[[V2-pending]]` may still change before the final runs.

| Item | Setting |
|---|---|
| Framework | nnU-Net v2, `3d_fullres`, plans `nnUNetPlans` \cite{isensee2021nnunet} |
| Stages / features per stage | 6 / (32, 64, 128, 256, 320, 320) |
| Patch size / batch size | 128 × 128 × 128 / 2 |
| Encoder initialization | VoComni \cite{wu2024voco,wu2024largescale}; 282/292 tensors loaded, 10 head tensors re-initialized |
| PFCR insertion points | 5 encoder skips (128³, 64³, 32³, 16³, 8³) |
| Feature proxy | 1×1×1 conv C → 1, initialized to the channel mean |
| Log-Gabor scales | S = 3; λ_init = 3.0 / 6.3 / 13.2 voxels (min. wavelength 3, multiplier 2.1); σ_onf = 0.55 |
| Reparameterization | ρ_s = 0.5·sigmoid(·); σ_s = 0.05 + 0.9·sigmoid(·); k = softplus(·), k_init = 2.0 |
| PC noise threshold | Kovesi estimate from the median amplitude of the finest scale \cite{kovesi1999phase}; [[V2-pending: hard vs soft]] |
| PC computation resolution | [[V2-pending: full resolution vs 2²⁰-voxel cap]] |
| Band grouping | Single-level 3D Haar DWT; low = LLL, mid = 3 subbands with one high-pass axis, high = 4 subbands with ≥ 2 high-pass axes |
| Residual scale | out = x + tanh(γ)·routed, γ_init = 0 |
| Tumor-prior head | Two 3×3×3 convs on the two shallowest skips; prior = softmax(logits)[tumor] |
| Loss | L_DiceCE (deep supervision) + 0.2·L_PC-rank + 0.05·L_budget + 0.1·L_prior |
| Deep-supervision weights | nnU-Net default: 1/2^i per output, lowest resolution weight 0, normalized to sum 1 |
| PC ranking margin | 0.2 |
| Budget targets (low, mid, high) | Inside tumor (0.20, 0.40, 0.40); outside tumor (0.70, 0.22, 0.08) |
| Prior loss | Class-weighted CE, tumor weight 4 |
| Auxiliary-loss schedule | Active from epoch 0, 5-epoch linear ramp |
| Optimizer | SGD, Nesterov, momentum 0.99 |
| Learning rate | 5 × 10⁻³ initial; multipliers encoder 0.3×, decoder 0.5×, PhASER 1.0×; poly decay (exponent 0.9) preserving the ratios |
| Weight decay | 3 × 10⁻⁵; PhASER scalar and reparameterized parameters excluded |
| Gradient clipping | Global norm 12 |
| Precision | AMP; FFTs in fp32 |
| Epochs × iterations | 200 × 250 [[TBD: confirm the same budget for all ablation groups]] |
| Foreground oversampling | 0.66 |
| Data augmentation | nnU-Net default \cite{isensee2021nnunet} [[TBD: confirm no trainer-specific override]] |
| Model selection | None; final-epoch weights |
| Inference | Sliding window, Gaussian weighting, tile step [[TBD: 0.5]]; no test-time augmentation |
| Post-processing | [[TBD: none, or nnU-Net largest-component rule if selected on validation]] |
| Random seeds | [[TBD: seed values]]; ≥ 3 seeds for key groups |
| Hardware / software | 1 × NVIDIA RTX 3090 24 GB; PyTorch 2.1, CUDA 11.8 |
| Training time per run | [[TBD]] h |

## 4.3 Evaluation protocol and statistics

We follow the recommendations of Metrics Reloaded \cite{maierhein2024metrics} and the pitfalls catalogued by Reinke et al. \cite{reinke2024pitfalls}. Metrics are computed on the original image grid of each case, after nnU-Net exports the prediction to that geometry. Distances are in millimeters and use the original voxel spacing.

**Cohorts.** *All27* denotes all 27 test cases. *Tumor24* denotes the 24 tumor-bearing test cases. Liver metrics and the false-positive rate use All27. Per-case tumor overlap and boundary metrics use Tumor24, because Dice is undefined when both reference and prediction are empty and is 0 for any false positive in a tumor-free case \cite{reinke2024pitfalls}. All27 tumor values, with these conventions stated, are given in the supplement.

**Overlap metrics.** For each case we compute the Dice similarity coefficient (DSC) and intersection over union (IoU). Liver means the whole liver (label > 0); tumor means label 2. Values are averaged over cases (per-case mean), the LiTS convention \cite{bilic2023lits}. We also report global tumor Dice, computed from voxel counts pooled over all cases. Global Dice weights large lesions more heavily. On one model and test set it can exceed the per-case value by about 12 points (85.72 vs. 73.63 in \cite{you2024pgcnet}). It is reported only to enable comparison with studies that use it.

**Lesion-wise metrics.** Individual lesions are 3D connected components of the tumor mask ([[TBD: 26-connectivity]]) in the reference and in the prediction. For each case we compute the IoU of every reference–prediction component pair. Pairs with IoU > 0.1 are candidate matches. A one-to-one assignment that maximizes the summed IoU over candidates (Hungarian algorithm) then defines true positives (TP). Unmatched reference components are false negatives (FN); unmatched predicted components are false positives (FP). This rule is stricter than the many-to-one merging used by the LiTS detection analysis \cite{bilic2023lits}, because a split lesion yields one TP plus FPs. We pool TP, FP and FN over the cohort and report lesion recall = TP/(TP + FN), precision = TP/(TP + FP), F1, and FP per scan (total FP / 27, All27). Results at an IoU > 0.5 matching threshold are given in the supplement. Instance-level reporting follows the argument that voxel-level overlap hides missed small lesions \cite{kofler2023blob}.

**Boundary metrics.** For tumor (Tumor24) we report the 95th-percentile symmetric Hausdorff distance (HD95, mm) and the normalized surface distance (NSD) at tolerance τ = [[TBD]] mm (2 mm is used in \cite{isensee2024revisited}). If a prediction contains no tumor in a Tumor24 case, HD95 is set to [[TBD: e.g., the image diagonal]] and NSD to 0. The number of such cases is reported.

**Strata.** Each reference lesion is assigned to a size stratum and a CNR stratum.
- *Size:* equivalent-sphere diameter d = (6V/π)^{1/3}, with V the component volume in mm³ ([[TBD: or maximal axial diameter]]). Strata: d < 10 mm, 10–20 mm, > 20 mm.
- *CNR:* CNR = |μ_lesion − μ_ring| / σ_ring, computed on HU intensities after resampling to 1.0 × 0.78 × 0.78 mm. The ring consists of liver voxels (label 1) lying 2–6 voxels outside the lesion. Restricting the ring to label 1 excludes other lesions and extra-hepatic tissue.
- The CNR tertile thresholds were frozen on the 518 training lesions: 1.2785 and 1.9208. Low CNR is < 1.2785, mid is 1.2785–1.9208, and high is > 1.9208. The training-lesion CNR has median 1.616 (range 0.022–5.373). The same thresholds are applied to validation, test and external lesions.

Lesion recall and the Dice of matched lesions are reported per stratum. FPs have no reference lesion and hence no reference size or CNR, so FP per scan is reported overall and by predicted-component size.

**Statistical analysis.** Key groups (A0, A1, A6, A7 and the external baselines; [[TBD: final list]]) are trained with at least 3 seeds. Other groups use [[TBD: 1]] seed. For each case, metrics are averaged over seeds before paired tests; per-seed results are reported in the supplement. Paired differences between two methods are summarized by the mean difference with a 95% confidence interval (CI) from a paired bootstrap over cases ([[TBD: 10,000]] resamples). Significance uses the two-sided Wilcoxon signed-rank test \cite{bilic2023lits}. The effect size is the matched-pairs rank-biserial correlation. For lesion-level quantities, the bootstrap resamples cases, not lesions, because lesions within a patient are correlated. Pre-specified comparisons are A6 vs. A0, A6 vs. A1, A6 vs. A7 and A6 vs. A8; their p-values are Holm-corrected. The primary endpoint is [[TBD: choose before test evaluation, e.g., recall of < 10 mm lesions at matched FP per scan, or per-case tumor Dice on Tumor24]].

**Model selection and test-set discipline.** All design, hyperparameter and post-processing decisions use the 21 validation cases only. Patch-level pseudo-Dice logged during training is not used for any decision, because on this dataset it does not track full-volume Dice. Every model is evaluated with its final-epoch weights. The 27 test cases and the external cohort are evaluated once, after the full protocol in this section was frozen ([[TBD: freeze date / commit hash]]). The mechanism analyses in Section 4.8 use validation cases only.

## 4.4 Compared methods

All methods are trained from scratch in our nnU-Net pipeline, on the same fold-0 split, with the same preprocessing, inference and evaluation code.

- **nnU-Net (random initialization)** \cite{isensee2021nnunet}: the default `PlainConvUNet` with the configuration of Table 2 but no pretraining and no PhASER modules. [[TBD: include if run]]
- **nnU-Net + VoComni (A0)**: the same network initialized from VoComni \cite{wu2024voco,wu2024largescale}. This is the direct baseline of PhASER-3D; the only difference is the PhASER modules and their losses.
- **U-Mamba** \cite{ma2024umamba}: [[TBD: U-Mamba_Bot and/or U-Mamba_Enc]], with the official nnU-Net-based implementation.
- **MedNeXt** \cite{roy2023mednext}: [[TBD: size (S/B/M/L) and kernel size (3 or 5)]], integrated in the same pipeline.
- **Swin UNETR** \cite{hatamizadeh2022swinunetr} (optional): [[TBD: include or not]].

Each external baseline uses its recommended optimizer and learning-rate schedule, the same epoch budget ([[TBD]]), and the largest patch size that fits in 24 GB ([[TBD per method]]). Strong, properly configured CNN baselines are hard to beat on 3D benchmarks \cite{isensee2024revisited}, so we did not shorten their training.

The external baselines start from random initialization, whereas PhASER-3D starts from VoComni. A comparison with them therefore mixes architecture and pretraining. The effect of the PhASER modules alone is measured by A6 vs. A0 (Section 4.6), and the effect of pretraining by A6 vs. A8.

**Why literature numbers are not copied.** Tumor Dice reported on LiTS ranges from about 65% to 90% across recent liver-tumor papers. Most of this spread follows from the protocol, not the method. Slice-level random splits place neighboring slices of one patient in both training and test sets. Some studies evaluate only tumor-containing slices. Others pool voxels into a global Dice. The same nnU-Net baseline is reported at 66.57% under a patient-level 3D protocol \cite{bian2026mambaclinix} and at 89.54% under a slice-level split restricted to tumor slices \cite{ma2025msffe}. Table S1 lists the split unit, test unit, Dice convention and reported values of these studies. We compare only against methods retrained under our protocol, and we refer to Table S1 to place our numbers in context. The best per-case tumor Dice in the LiTS challenge (0.674, 0.702 and 0.739 across three rounds \cite{bilic2023lits}) uses a different, 70-case hidden test set. It is a reference point only, not a comparison.

## 4.5 Comparison results

Table 3 reports the comparison on the internal test set (part a) and the external test set (part b). Methods are grouped by architecture family, following \cite{bian2026mambaclinix}. Fig. 4 shows qualitative results for [[TBD: number]] test cases, chosen to span the three CNR tertiles and the three size strata. Case selection rule: [[TBD: e.g., the case closest to the median per-case Dice of A0 within each stratum]].

[[TBD: results narrative. Report per-case tumor Dice of PhASER-3D vs. A0 with paired 95% CI and Wilcoxon p; then the strongest external baseline; then lesion recall, FP per scan and HD95. State explicitly where differences are within the CI.]]

[[CLAIM-PENDING: depends on A6 vs A0/A1/A7]] Any statement that PhASER-3D improves over A0 must cite the paired CI, not only the means.

<!-- Expected reading: The hypothesis is supported if A6 exceeds A0 in per-case tumor Dice (paired CI excluding 0, consistent sign across seeds) without higher FP per scan, and its gap to external baselines stays when they are compared at matched pretraining status. It is refuted if the A6 − A0 CI includes 0 on both internal and external test sets, or if gains appear only in global Dice. -->

**Table 3.** Comparison on the LiTS internal test set (a) and the external test set (b). Liver: All27, whole liver (label > 0). Tumor overlap and boundary: Tumor24. Lesion metrics: IoU > 0.1 one-to-one matching, pooled; FP/scan over All27. Values are mean ± SD over cases, averaged over seeds. "Params" counts trainable parameters. Bold marks the best value only where the paired CI vs. the runner-up excludes 0.

*(a) LiTS internal test set (27 cases; 24 tumor-bearing)*

| Family | Method | Params (M) | Liver DSC (%) | Liver IoU (%) | Tumor DSC (%) | Tumor IoU (%) | Tumor global DSC (%) | Lesion recall (%) | Lesion precision (%) | FP/scan | Tumor HD95 (mm) | Tumor NSD (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN-based | nnU-Net (random init.) \cite{isensee2021nnunet} | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| CNN-based | nnU-Net + VoComni (A0) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| CNN-based | MedNeXt \cite{roy2023mednext} | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| Transformer-based | Swin UNETR \cite{hatamizadeh2022swinunetr} (optional) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| Mamba-based | U-Mamba \cite{ma2024umamba} | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| Frequency-aware | PhASER-3D (ours, A6) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| | Δ A6 − A0 [95% CI] | – | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |

*(b) External test set ([[TBD: cohort]], [[TBD]] cases)*

| Family | Method | Liver DSC (%) | Tumor DSC (%) | Tumor IoU (%) | Lesion recall (%) | Lesion precision (%) | FP/scan | Tumor HD95 (mm) | Tumor NSD (%) |
|---|---|---|---|---|---|---|---|---|---|
| CNN-based | nnU-Net + VoComni (A0) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| CNN-based | MedNeXt | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| Mamba-based | U-Mamba | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| Frequency-aware | PhASER-3D (ours, A6) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |

## 4.6 Ablation study

The ablation adds one component at a time, in the incremental style of \cite{bian2026mambaclinix}, and adds two control groups. All groups share the backbone, data, budget and inference settings of Table 2. Table 4 lists what each group adds and the question it answers.

- **A0** — nnU-Net + VoComni, no PhASER module. Baseline.
- **A1** — skip gate driven by the feature gradient magnitude |∇f|: out = x·(1 + tanh(γ)·σ(a·(g − b))), where g is the normalized |∇f| of the feature proxy and a, b are learnable. *Amplitude control*: does any boundary-like gate help?
- **A2** — the same gate with PC in place of |∇f|. *Phase vs. amplitude* (A2 vs. A1) at equal structure.
- **A3** — PFCR without the tumor prior: route = f(PC). *Routing vs. gating* (A3 vs. A2).
- **A4** — PFCR + tumor prior (router input [PC, prior]; prior head trained with L_prior). *Semantic conditioning* (A4 vs. A3).
- **A5** — A4 + L_budget. *Band-budget constraint* (A5 vs. A4).
- **A6** — A5 + L_PC-rank: full PhASER-3D. *Ranking supervision of the PC operator* (A6 vs. A5).
- **A7** — A6 with PC spatially shuffled inside each sample before the router ([[TBD: shuffle scheme, e.g., a random voxel permutation redrawn every forward pass]]). Parameters, FLOPs and the value distribution of PC are unchanged; only its spatial correspondence with the image is destroyed. *Capacity control* (A7 vs. A6): is the gain due to phase evidence or to extra parameters and regularization?
- **A8** — A6 trained from random initialization. *Pretraining control* (A8 vs. A6 and A8 vs. nnU-Net from scratch).

Because A3 → A6 add one loss term at a time, the same table also serves as the loss ablation. Following \cite{bian2026mambaclinix}, it reports both DSC and recall, since an auxiliary term can trade one for the other.

[[TBD: results narrative, in the order A1 vs. A0, A2 vs. A1, A3 vs. A2, A4 vs. A3, A5 vs. A4, A6 vs. A5, A7 vs. A6, A8 vs. A6. For each contrast, state the paired Δ tumor DSC with 95% CI and the Δ lesion recall.]]

<!-- Expected reading: The phase-routing hypothesis is supported if (i) A2 > A1 and A6 > A1 with CIs excluding 0, (ii) A7 falls back towards A0/A1 (so the gain is not capacity), and (iii) each of A4, A5, A6 adds a non-negative step. It is refuted if A1 ≈ A6 (amplitude gating explains the gain), if A7 ≈ A6 (capacity/regularization explains it), or if the gain disappears in A8 relative to nnU-Net from scratch (pretraining interaction). -->

**Table 4.** Incremental ablation on the LiTS internal test set. ✓ = component present. "Rank" = L_PC-rank. FLOPs per 128³ forward pass. DSC and recall in %. Δ is the paired difference to A0 with 95% bootstrap CI.

| ID | Gate signal | PFCR routing | Prior | L_budget | Rank | VoComni | Params (M) | FLOPs (G) | Tumor DSC | Tumor IoU | Lesion recall | Lesion F1 | FP/scan | Δ DSC vs. A0 [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | – | – | – | – | – | ✓ | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | – |
| A1 | \|∇f\| | – | – | – | – | ✓ | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| A2 | PC | – | – | – | – | ✓ | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| A3 | PC | ✓ | – | – | – | ✓ | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| A4 | PC | ✓ | ✓ | – | – | ✓ | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| A5 | PC | ✓ | ✓ | ✓ | – | ✓ | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| A6 (full) | PC | ✓ | ✓ | ✓ | ✓ | ✓ | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| A7 (shuffled PC) | PC, shuffled | ✓ | ✓ | ✓ | ✓ | ✓ | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| A8 (scratch) | PC | ✓ | ✓ | ✓ | ✓ | – | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |

Seeds per group: [[TBD]]. Validation-set results for all groups are given in the supplement.

## 4.7 Stratified analysis (lesion size and CNR)

Aggregate Dice is dominated by large lesions. A method can therefore change small or low-contrast lesions without moving the mean. Table 5 reports lesion recall and matched-lesion Dice per size and CNR stratum, using the frozen definitions of Section 4.3. The CNR distribution matters here. On the training lesions the median CNR is 1.616, so lesions below CNR 2 are the majority of the data, not rare exceptions.

We test the interaction directly. For each stratum, we compute the paired difference in lesion recall between A6 and A1, and between A6 and A0. We then estimate the difference of these gains between the low and high CNR tertiles, with a case-level bootstrap CI. The same is done for the < 10 mm vs. > 20 mm size strata.

[[TBD: results narrative. Give n lesions per stratum on the test set, then recall per method and stratum, then the low-minus-high interaction estimate with CI.]]

<!-- Expected reading: The low-contrast hypothesis is supported if the A6 − A1 recall gain is positive in the low-CNR tertile and larger there than in the high-CNR tertile (interaction CI excluding 0), with no increase in FP per scan. It is refuted if the gain is zero in the low tertile or concentrated in the high tertile. Section 4.8 shows that the untrained PC operator gives no low-CNR advantage, so a refutation here is plausible and must be reported as such. A gain confined to the < 10 mm stratum without a CNR pattern would point to a size effect, not a contrast effect. -->

**Table 5.** Stratified lesion-level results on the LiTS internal test set. Recall (%) and mean Dice of matched lesions (%). CNR tertile thresholds frozen on 518 training lesions (1.2785 / 1.9208).

| Stratum | n lesions (test) | A0 recall | A1 recall | A6 recall | Best ext. baseline recall | A0 matched DSC | A1 matched DSC | A6 matched DSC | Δ recall A6 − A1 [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| d < 10 mm | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| 10 ≤ d ≤ 20 mm | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| d > 20 mm | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| CNR < 1.2785 (low) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| 1.2785 ≤ CNR ≤ 1.9208 (mid) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| CNR > 1.9208 (high) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| Interaction: low − high CNR | – | – | – | – | – | – | – | – | [[TBD]] |
| Interaction: < 10 mm − > 20 mm | – | – | – | – | – | – | – | – | [[TBD]] |

## 4.8 Mechanism analysis

This section asks what the PC evidence does, separately from the segmentation score. Fig. 3 characterizes the PC operator itself, before training. Fig. 5 examines what training changes in the full model. All analyses use synthetic volumes or the 21 validation cases; no test case is used.

**Fig. 3: the PC operator before training (completed).** All numbers below were computed with the PC operator at its initial parameter values (S = 3, minimum wavelength 3, multiplier 2.1, σ_onf = 0.55, no resolution cap). The operator was applied directly to HU intensities, not to network features. They characterize the operator, not the trained network. The comparison operator is the gradient magnitude |∇f|, the same operator as in A1.

*Synthetic setup.* Volumes of 160³ isotropic voxels follow x = 60 + A·obj + σ·n, with n white Gaussian noise. Three geometries were used: a thick slab (half-thickness 60 voxels), a ball (radius 60), and a slab with a 3-voxel soft transition. Interior and exterior reference regions lie at least 30 voxels from the boundary, beyond the support of the largest filter. Five noise seeds were used. Boundary localization was measured by the boundary-to-interior response ratio and the area under the ROC curve (AUC) separating boundary from interior voxels.

*(i) Contrast invariance holds.* Without noise, raising the contrast A from 5 to 40 HU changed PC by at most 3.0 × 10⁻³ (slab), 8.6 × 10⁻³ (soft slab) and 2.3 × 10⁻² (ball) at any voxel. Over the same range, boundary |∇f| divided by A stayed constant (0.185, 0.068 and 0.206 for slab, soft slab and ball). Thus PC(a·x) = PC(x) holds in the implementation, while |∇f| scales linearly with contrast.

*(ii) Without noise, PC grades inside vs. outside rather than localizing the edge.* On a noise-free slab (A = 20 HU), a profile across the edge gave PC = 0.644 at the slab center, 0.996 at the edge and 0.911 outside. The edge-to-interior contrast of PC was 1.28×, against 4.9 × 10⁴× for |∇f|. On large homogeneous structures, local phase is coherent over whole half-spaces.

*(iii) With realistic noise, the noise threshold silences PC at low CNR.* With σ = 5 HU, Kovesi's threshold \cite{kovesi1999phase} was T ≈ 9.8. The volume-mean summed amplitude Σ_s A_s was only 4.9–8.5 for CNR between 0.25 and 16. Because PC = (‖Σ_s v_s‖ − T)₊ / (Σ_s A_s + ε) and ‖Σ_s v_s‖ ≤ Σ_s A_s, PC was zero at every voxel for CNR ≤ 1. At the edge, PC became non-zero only at higher contrast (0.047 at CNR 4; 0.547 at CNR 16). Correspondingly, for CNR ≤ 1.5 both PC and |∇f| had boundary-vs-interior AUC between 0.500 and 0.506, i.e., no localization ability. On the slab, the AUC at CNR 8 was 0.844 for PC and 0.614 for |∇f|; at CNR 16 it was 1.000 and 0.804. The PC boundary ratio was never materially below that of |∇f| (≥ 0.98×). Its advantage, however, grew with CNR instead of shrinking.

*(iv) Real lesions show the same pattern.* We cropped 91 lesions (≥ 50 voxels) from the 21 validation cases, resampled with their CT to 1.0 × 0.78 × 0.78 mm in HU. Of these, 87 were analyzed (4 exceeded the crop size). We compared each operator's mean response on the lesion boundary shell with its mean response on the peritumoral liver ring (2–6 voxels outside, label 1). In the low-CNR tertile (n = 36, median CNR 0.959), the median boundary-to-ring ratio was 0.651 for PC and 0.967 for |∇f|. Both operators responded less at the lesion boundary than in the surrounding liver; the effect was stronger for PC. In the high-CNR tertile (n = 22), the ratios were 1.678 for PC and 1.111 for |∇f|. PC had the higher ratio for 9 of 36 low-CNR lesions and 15 of 22 high-CNR lesions. The boundary–ring separation correlated with CNR for both operators (Spearman ρ = 0.530 for PC, 0.708 for |∇f|).

*Reading of Fig. 3.* With its initial parameters, the PC operator is contrast-invariant but not noise-invariant. Under CT-like noise it abstains at low CNR and responds selectively at strong edges. It therefore does not, by itself, supply low-contrast boundary evidence on raw intensities. Any low-CNR benefit of PhASER-3D would have to come from what training changes: learned scales and noise multiplier, PC computed on learned feature proxies rather than HU, and the tumor prior. Figs. 3 (trained panel) and 5 test this. [[V2-pending: if the V2 noise threshold is changed (e.g., soft threshold), repeat (iii) for the V2 operator and report both.]] Caveat: the synthetic noise was white and additive, whereas CT noise is spatially correlated and intensity-dependent. The direction of the effect follows from the threshold mechanism, but the exact CNR at which PC switches on may differ.

**Fig. 3 (trained panel) and Fig. 5: what training changes ([[TBD]]).** For the final A5 and A6 models (all seeds), on the validation cases:
1. *Learned filters.* Center wavelength λ_s and bandwidth σ_s per log-Gabor scale and per skip level, against the initial 3.0 / 6.3 / 13.2 voxels; learned noise multiplier k. We report the relative shift |Δλ|/λ_init and the spread of λ across the five skip levels. Comparing A6 with A5 isolates the effect of L_PC-rank.
2. *PC on features.* The boundary-to-ring PC ratio and the boundary-vs-ring AUC per CNR tertile, computed on the feature proxy of each skip of the trained network. This is the trained counterpart of Fig. 3(iv).
3. *Route maps.* Mean route (low, mid, high) inside tumor, in liver parenchyma and in background, per skip level. Route selectivity is the absolute difference between the mean tumor route and the mean background route. Spatial variability is the coefficient of variation of the route over the patch. Distance to the budget targets is the L1 distance to (0.20, 0.40, 0.40) inside tumor and (0.70, 0.22, 0.08) outside. The route-statistics code is re-validated on synthetic inputs with known routes before use ([[TBD: unit test description]]).
4. *Residual scale.* tanh(γ) per skip level. A value near 0 means the module has little effect on the skip.

[[TBD: results narrative.]]

<!-- Expected reading: The mechanism claim is supported if, after training, (a) PC on features separates boundary from ring in the low-CNR tertile (ratio > 1, AUC > 0.5 with CI), unlike the untrained operator on HU; (b) route selectivity is clearly non-zero and routes approach the budget targets; and (c) A6 filters differ from A5 filters. It is refuted if routes stay spatially near-constant (PFCR acting as a fixed band-pass filter per level), if selectivity is near zero, or if A5 and A6 filters coincide. In that case the paper should describe PFCR as a learned per-level frequency re-weighting, not as content-dependent routing. -->

## 4.9 Inference-time interventions and controlled degradation

**Interventions.** Interventions change one internal signal of a trained model at inference time and measure the change in output. They are applied to the final A6 models (all seeds) on the validation cases and then, once, on the test cases. Table 6 lists them:
- *Zero PC at level ℓ:* the PC map entering the router at skip level ℓ ∈ {1,…,5} is set to 0; other levels are unchanged. [[TBD: also per log-Gabor scale s ∈ {1, 2, 3}?]] Note that PC = 0 is also what the noise threshold produces at low CNR.
- *Zero PC at all levels.*
- *Uniform route:* the route is fixed to (1/3, 1/3, 1/3) at every voxel and level, so band mixing no longer depends on content.
- *Zero prior:* the tumor prior entering every router is set to 0.
- *PFCR off:* the routed residual is removed (out = x) at every level. This bounds the total contribution of the modules.

For each intervention we report the change in tumor DSC, lesion recall and FP per scan relative to the unmodified model. We also report the per-lesion utility U_j = L_j(intervened) − L_j(full), where L_j is the soft-Dice loss restricted to a box around lesion j. Positive U_j means the removed signal helped lesion j. U_j is summarized per CNR tertile. These interventions measure the contribution of a signal *within a fixed trained network*. They are not causal effects in a clinical sense, and they are not Shapley values.

**Controlled degradation.** To test whether the method helps as lesion contrast falls, we reduce lesion contrast in real CT and keep the labels fixed. For each reference lesion j with ring mean μ_ring,j (Section 4.3), voxel intensities inside the lesion are moved toward μ_ring,j:

$$x'_v = x_v - (1-\alpha)\,(\mu_{\mathrm{lesion},j} - \mu_{\mathrm{ring},j}), \qquad v \in \text{lesion } j,\quad \alpha \in \{1, 0.75, 0.5, 0.25\}.$$

This shift leaves the lesion's internal texture and the ring statistics unchanged, so each lesion's CNR becomes exactly α·CNR. [[TBD: confirm mean shift vs. multiplicative scaling x' = μ_ring + α(x − μ_ring); the latter also shrinks internal texture by α.]] To avoid an artificial step at the mask border, the shift is weighted by a mask smoothed with a Gaussian of σ = [[TBD: 1]] voxel. Degradation is applied in HU, before nnU-Net normalization. It is applied to A0, A1, A6, A7 and the external baselines, on the validation cases and then on the test cases. Fig. 6 plots lesion recall against the degraded lesion CNR for each method. The key quantity is the interaction between method and α: does the A6 − A1 recall gap widen as α decreases? It is estimated with a case-level bootstrap CI.

[[TBD: results narrative.]]

<!-- Expected reading (interventions): PC-based routing is used by the network if zeroing PC or forcing a uniform route lowers lesion recall or Dice beyond seed-to-seed variation, with positive U_j concentrated in the strata where the method gains. If outputs are unchanged under zero PC or uniform route, the modules act as fixed filters and the PC evidence is not used. Expected reading (degradation): the low-contrast claim is supported if A6 loses recall more slowly than A1 and A0 as α decreases (interaction CI excluding 0). It is refuted if the curves are parallel or if A6 degrades faster. -->

**Table 6.** Inference-time interventions on the final A6 models (LiTS internal test set; mean over seeds). Δ relative to the unmodified model; 95% CI from a case-level bootstrap.

| Intervention | Δ Tumor DSC (%) [95% CI] | Δ Lesion recall (%) [95% CI] | Δ FP/scan | Median U_j, low CNR | Median U_j, mid CNR | Median U_j, high CNR |
|---|---|---|---|---|---|---|
| None (A6) | 0 | 0 | 0 | – | – | – |
| Zero PC, level 1 (128³) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| Zero PC, level 2 (64³) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| Zero PC, level 3 (32³) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| Zero PC, level 4 (16³) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| Zero PC, level 5 (8³) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| Zero PC, all levels | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| Uniform route (1/3, 1/3, 1/3) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| Zero prior | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| PFCR off (out = x) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |

## 4.10 Computational cost

Table 7 reports model size and cost. Parameters are trainable parameters. FLOPs are counted for one forward pass on a 128³ patch with batch size 1, using [[TBD: profiler]]. Hook-based profilers often skip FFT and DWT operations. The FLOPs of the PC operator (FFTs, Riesz filtering) and of the Haar transforms are therefore counted analytically and added: [[TBD: confirm]]. Training time is wall-clock time for the full schedule (Table 2) on one RTX 3090. Inference time is the mean ± SD per case over the 27 test cases. It covers sliding-window prediction without test-time augmentation, excludes disk I/O and preprocessing ([[TBD: confirm]]), and is measured after one warm-up case. Peak GPU memory during training is also reported.

[[TBD: results narrative. State the PhASER overhead over A0 in parameters, FLOPs, training time and inference time, in absolute and relative terms.]]

<!-- Expected reading: The method is practical if its overhead over A0 is small in parameters (the modules add a few small convolutions per level) and moderate in time, with the FFT-based PC at the 128³ level expected to dominate. The cost argument weakens if inference time grows by a large factor relative to A0 without a matching accuracy gain in Tables 3–5. -->

**Table 7.** Computational cost. FLOPs per 128³ forward pass (batch 1). Times on one NVIDIA RTX 3090 (24 GB).

| Method | Params (M) | FLOPs (G) | Peak train memory (GB) | Training time per epoch (min) | Total training time (h) | Inference time per case (s) |
|---|---|---|---|---|---|---|
| nnU-Net + VoComni (A0) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| A1 (gradient gate) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| PhASER-3D (A6) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| MedNeXt | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| U-Mamba | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| Swin UNETR (optional) | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |

## Open questions for the authors

- **Code vs. V2 specification.** The local trainer (`nnUNetTrainer_phaser.py`, `phaser/modules.py`) still has the pre-V2 behaviour. (a) Budget, lesion-weight and small-head losses are gated to "Phase B" (epoch ≥ 60) with a 50-epoch ramp; only PC ranking starts at epoch 0 with a 5-epoch ramp. (b) The small-head loss has weight 0.1 and tumor-voxel weight 4, as in the brief, but it is also Phase-B gated. (c) PFCR applies `prior.sigmoid()` to an already-sigmoided prior (double sigmoid). (d) No weight-decay exclusion for PhASER scalars. (e) `pc_max_voxels = 2²⁰` by default. (f) Spectral, boundary, lesion-weight and focal-Tversky terms exist and were active in the old run. Table 2 follows the brief (V2). Please confirm that V2 removes (f) and fixes (a)–(e).
- **Router architecture.** The brief writes route = softmax(Conv3(ReLU(Conv3(·)))). The code uses Conv3×3×3(1 or 2 → 8) → ReLU → **Conv1×1×1**(8 → 3). Which is V2? Table 2 does not state the kernel of the last layer.
- **Epoch budget for ablations.** The brief says 200 × 250 for all runs; `validation_plan.txt` proposes 100 epochs for the ablation matrix. Which budget applies to A0–A8 and the external baselines?
- **Lesion count criterion.** Are the 518 training lesions connected components of ≥ 50 voxels (as for the 91 validation lesions), and with which connectivity?
- **Lesion matching threshold.** The brief uses IoU > 0.1; `validation_plan.txt` §10.3 proposes IoU > 0.5 with sensitivity analysis. I made 0.1 primary and 0.5 supplementary. Confirm, and confirm Hungarian one-to-one matching instead of LiTS-style merging.
- **Lesion diameter definition.** Equivalent-sphere diameter or maximal axial diameter (RECIST-like)? This changes the < 10 mm stratum.
- **NSD tolerance τ and HD95 for empty predictions.** Please fix τ (2 mm as in nnU-Net Revisited?) and the HD95 value when a Tumor24 prediction is empty.
- **Primary endpoint.** It must be fixed before the test evaluation (candidates in `validation_plan.txt` §8). Which one?
- **Contrast-invariance number in the brief.** The brief quotes max |ΔPC| ≤ 8.6 × 10⁻³. The Step 2 report gives 3.0 × 10⁻³ (slab), 8.6 × 10⁻³ (soft slab) and 2.3 × 10⁻² (ball). I reported all three; the brief's bound holds only for the slab geometries.
- **Step 2 operator parameters.** Please confirm that the synthetic threshold analysis (report §3.2, T ≈ 9.8) used the initial k = 2.0, as the caller states, and not the learned k ≈ 1.76 of the old checkpoint. The report text mentions the learned value next to it.
- **Narrative risk.** The completed Step 2 evidence (Fig. 3) does not support "PC is more robust than gradient magnitude at low CNR" for the untrained operator. I wrote 4.8 to report this and to place the burden on the trained-model analyses. The Introduction and Method sections should not claim a low-CNR advantage before Tables 5 and 6 and Fig. 6 are in.
- **Degradation operator.** Mean shift (keeps lesion texture, CNR scales exactly by α) or multiplicative scaling toward the ring mean (also shrinks lesion texture)? And the smoothing σ of the blending mask.
- **External cohort.** TCIA Colorectal-Liver-Metastases or HCC-TACE-Seg? Does it provide liver labels, or tumor only? This decides the columns of Table 3(b).
- **Baselines.** Which U-Mamba variant (Bot/Enc), which MedNeXt size and kernel, whether Swin UNETR and a random-init nnU-Net are included, and whether external baselines get the same epoch budget or their recommended one.
- **Global Dice.** `reproduce_metrics.py` does not yet compute global (pooled) tumor Dice; it is needed for Table 3.
- **Post-processing.** Is any post-processing (e.g., largest liver component, removal of tumor outside the liver) selected on validation? Table 2 currently says [[TBD]].
- **3D-IRCADb overlap.** The text states overlap is "likely and cannot be excluded", based on IRCAD contributing 20 LiTS training scans and 3D-IRCADb-01 having 20 volumes. If you have verified it (e.g., by image hashes), the sentence can be made definitive.
