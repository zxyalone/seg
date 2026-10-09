# Writing brief — PhASER-3D manuscript (shared by all section writers)

Target journal: **Biomedical Signal Processing and Control (BSPC, Elsevier)**. Language: **English** (American spelling). Draft format: Markdown, one file per section under `manuscript/sections/`, math in LaTeX (`$...$`, `$$...$$`), citations as `\cite{key}` using ONLY the keys in §7 (add a new key only if unavoidable, and list it under "Open questions" at the end of your file).

## 1. Hard rules

1. **Never invent results.** Any number we do not yet have is written as a placeholder `[[TBD: what goes here]]`. Tables of results are written as shells with `[[TBD]]` cells.
2. Numbers you MAY use are listed in §5. Literature numbers must come from `文献/文献对比表.md`.
3. Claims about the method's benefit must be phrased so they stay true whatever the ablation shows, or be marked `[[CLAIM-PENDING: depends on A6 vs A0/A1/A7]]`.
4. Write only your own file(s). Do not edit code, other sections, or anything outside `manuscript/sections/`. No git, no training, no network calls needed.
5. End every file with a section `## Open questions for the authors` (bullets).
6. Style: concise, specific, no hype words ("remarkable", "superior" without numbers). Sentences < 30 words where possible. Define every acronym once.

## 2. Structure (modelled on MambaClinix, BSPC 2026; intro subsections modelled on T-MPEDNet, BSPC 2025)

- Abstract (written last by the lead author — do not write)
- 1 Introduction — 1.1 Clinical problem and challenges · 1.2 Research trajectory (CNN → Transformer/Mamba → frequency-aware networks) · 1.3 Gaps (amplitude-based boundary cues fail at low contrast; frequency modules not conditioned on content; evaluation protocols not comparable) · 1.4 Contributions (numbered, 3–4 items) · 1.5 Outline
- 2 Related Work — 2.1 Liver tumor segmentation · 2.2 Frequency- and wavelet-domain segmentation networks · 2.3 Phase congruency and the monogenic signal · 2.4 Evaluation protocols on LiTS
- 3 Method — 3.1 Overview · 3.2 Monogenic phase congruency with learnable log-Gabor scales · 3.3 Phase–Frequency Coherence Routing (PFCR) · 3.4 Tumor-prior head · 3.5 Training objective
- 4 Experiments and Results — 4.1 Datasets · 4.2 Implementation details · 4.3 Evaluation protocol and statistics · 4.4 Compared methods · 4.5 Comparison results · 4.6 Ablation study · 4.7 Stratified analysis (lesion size, CNR) · 4.8 Mechanism analysis · 4.9 Inference-time interventions and controlled degradation · 4.10 Computational cost
- 5 Discussion — comparability with literature, what the mechanism analysis shows, limitations
- 6 Conclusion
- Supplementary S1 — literature protocol table, reproducibility checklist

## 3. The method (PhASER-3D, "V2" configuration used for all new experiments)

Backbone: 3D nnU-Net `PlainConvUNet` (6 stages, features 32/64/128/256/320/320, patch 128³, batch 2, deep supervision), encoder initialized from **VoComni** pretrained weights (VoCo family, \cite{wu2024voco,wu2024largescale}); 282/292 keys loaded, the 10 segmentation-head keys re-initialized (21 → 3 classes).

PFCR is inserted on each of the 5 non-bottleneck encoder skips (resolutions 128³, 64³, 32³, 16³, 8³):
- **Proxy**: 1×1×1 conv C→1, initialized to the channel mean.
- **Phase congruency (PC)**: monogenic signal via Riesz transform \cite{felsberg2001monogenic}; S = 3 learnable log-Gabor scales, center wavelengths initialized at 3.0, 6.3, 13.2 voxels (min wavelength 3, multiplier 2.1), bandwidth σ_onf = 0.55; reparameterized ρ_s = 0.5·sigmoid(·), σ_s = 0.05 + 0.9·sigmoid(·). PC = (‖Σ_s v_s‖ − T)_+ / (Σ_s A_s + ε), the PC₁ form used by FSIM \cite{zhang2011fsim}, with Kovesi's noise threshold T \cite{kovesi1999phase} estimated from the median amplitude of the finest scale, noise multiplier k = softplus(learnable), init 2.0. FFT in fp32. Contrast invariance: PC(a·x) = PC(x) for a > 0.
- **Routing**: route = softmax(Conv3(ReLU(Conv3([PC, prior])))) ∈ R³ per voxel (low / mid / high); last conv zero-initialized → uniform route at init.
- **Band mixing**: single-level 3D Haar DWT gives 8 subbands; band group *low* = LLL, *mid* = the 3 subbands with one high-pass axis, *high* = the 4 subbands with ≥2 high-pass axes; each group is scaled by its route weight (route resized to subband resolution) and inverted by IDWT.
- **Residual**: out = x + tanh(γ)·routed, γ initialized to 0 → exact identity at initialization (protects pretrained features).
- **Tumor-prior head** (Small-Tumor Head): two 3×3×3 convs on the two shallowest skips → 2-class logits; prior = softmax(logits)[tumor] (V2 fix; earlier code applied sigmoid twice — do not describe the old behaviour in the paper body).
- **Losses (V2)**: L = L_DiceCE (deep supervision, nnU-Net weights) + 0.2·L_PC-rank + 0.05·L_budget + 0.1·L_prior.
  - L_PC-rank: margin ranking, margin 0.2: relu(margin − (mean PC on tumor boundary shell − mean PC on {eroded tumor interior ∪ peritumoral ring 2–3 voxels})), averaged over skips/samples containing tumor. Rationale: PC is a generic structure detector; regressing it onto a lesion-boundary mask would penalize correct responses on vessels and capsule.
  - L_budget: L1 distance of mean route inside tumor to (0.20, 0.40, 0.40) and outside tumor to (0.70, 0.22, 0.08).
  - L_prior: class-weighted CE (tumor voxels weight 4) of the prior head vs. tumor mask.
  - All auxiliary losses active from epoch 0 with a 5-epoch linear ramp. PhASER scalar/reparameterized parameters are excluded from weight decay.
- Optimizer: SGD, Nesterov, momentum 0.99, initial LR 5e-3 with per-group multipliers (encoder 0.3×, decoder 0.5×, PhASER 1.0×), poly decay (exponent 0.9) preserving the ratios, weight decay 3e-5, grad-norm clip 12, AMP. 200 epochs × 250 iterations, foreground oversampling 0.66. **Final-epoch weights are reported; no checkpoint selection.** Inference: sliding window, no test-time augmentation.
- Mark anything that may still change as `[[V2-pending]]` (e.g., soft vs. hard noise threshold).

Code lives in `nnUNet/nnunetv2/training/nnUNetTrainer/` (`nnUNetTrainer_phaser.py`, `phaser/{unet,modules,losses,dim_utils}.py`). Read it to get details right.

## 4. Data and protocol

- LiTS training set (131 contrast-enhanced CT, 7 centers) \cite{bilic2023lits}; LiTS = MSD Task03 \cite{simpson2019msd}. Our split (nnU-Net `Dataset102_LiTS`, fold 0): **83 train / 21 validation / 27 test**, patient-level, no overlap. 24 of the 27 test cases contain tumor (LiTS_032, 106, 115 do not).
- Resampled to 1.0 × 0.78 × 0.78 mm (z, y, x); nnU-Net CT normalization.
- **All model and post-processing choices use the 21 validation cases only.** The 27 test cases are evaluated once at the end. External test: TCIA Colorectal-Liver-Metastases or HCC-TACE-Seg `[[TBD: which]]`. Note: 3DIRCADb is NOT independent of LiTS (LiTS training contains 20 IRCAD scans) — do not use it as external validation.
- Metrics: per-case Dice and IoU for liver (whole liver, label > 0) and tumor (label 2), averaged over cases (LiTS convention); lesion-wise recall / precision / F1 (connected components, IoU > 0.1 matching), FP per scan, HD95, NSD; strata by lesion diameter (<10, 10–20, >20 mm) and lesion–liver CNR tertiles.
- CNR = |μ_lesion − μ_ring| / σ_ring, ring = liver voxels 2–6 voxels outside the lesion; tertile thresholds frozen on the 518 training lesions: **1.2785 / 1.9208**; training-lesion CNR median **1.616** (range 0.022–5.373).
- Statistics: ≥3 seeds for key groups; paired bootstrap 95% CI and Wilcoxon signed-rank on per-case values; effect sizes reported, not only p-values.
- Ablation groups: A0 nnU-Net + VoComni (no PhASER); A1 gradient-magnitude skip gate (amplitude control); A2 PC skip gate without routing; A3 PFCR without prior; A4 PFCR + prior; A5 A4 + budget; A6 full PhASER-3D (A5 + PC ranking); A7 A6 with spatially shuffled PC (capacity control); A8 A6 trained from scratch.

## 5. Numbers you may use now

- LiTS challenge best per-case tumor Dice: 0.674 (ISBI 2017), 0.702 (MICCAI 2017), 0.739 (MICCAI 2018); best liver 0.963 \cite{bilic2023lits}.
- Literature protocol numbers: see `文献/文献对比表.md` (e.g., PGC-Net: global Dice 85.72 vs per-case 73.63 on the same model; MSFFE-Net vs nnU-Net under slice-level split 89.61 vs 89.54; MambaClinix tumor DSC 71.37 vs nnU-Net 66.57).
- Mechanism (Step 2, `step2_mechanism/report_step2_mechanism.md`), usable as motivation/analysis: in noise-free synthetic volumes PC is contrast-invariant (max |ΔPC| ≤ 8.6e-3 when contrast changes 5→40 HU) while gradient magnitude scales linearly; with σ = 5 HU white noise, Kovesi's threshold suppresses PC almost everywhere at CNR ≤ 1. Caveat: these were computed with the un-trained operator on HU intensities, not on network features.
- Do NOT use the old Copy2 / retrain numbers (0.7029, 0.7467) as results of the proposed method; they came from a different code version.

## 6. Figures and tables (reference by these labels)

Fig. 1 overall architecture · Fig. 2 PFCR module · Fig. 3 PC vs gradient response vs CNR (synthetic + real) · Fig. 4 qualitative cases · Fig. 5 mechanism (learned filters, route maps) · Fig. 6 recall vs CNR under controlled degradation.
Table 1 datasets · Table 2 implementation/configuration · Table 3 comparison results · Table 4 ablation · Table 5 stratified results · Table 6 interventions · Table 7 cost · Table S1 literature protocols.

## 7. Citation keys (bibliography in `manuscript/refs.bib`)

Benchmarks/data: bilic2023lits, simpson2019msd · Baselines: isensee2021nnunet, isensee2024revisited, ma2024umamba, roy2023mednext, hatamizadeh2022swinunetr, li2018hdenseunet · Pretraining: wu2024voco, wu2024largescale · Phase/frequency: kovesi1999phase, felsberg2001monogenic, zhang2011fsim, li2020wavecnn, jiang2025texlivernet, han2025frequency · Evaluation: maierhein2024metrics, reinke2024pitfalls, kofler2023blob · BSPC liver tumor papers: ma2025msffe, bian2026mambaclinix, cheng2024egunetr, liu2025okmil, wang2025rioUNeXt, wu2025tcanet, zhu2025bppnet, zheng2025abedice, hu2024msml, qi2024addunet, peng2024spidernet, you2024pgcnet, raghaw2025tmpednet

PDFs are in `文献/` (index: `文献/README.md`). Read the ones you cite; describe them accurately.
