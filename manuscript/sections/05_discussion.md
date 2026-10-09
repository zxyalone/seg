# 5. Discussion

## 5.1 Comparability with published LiTS results

Recent studies report LiTS tumor Dice from about 65% to 90% (Table S1). Most of this spread comes from three protocol choices rather than from the architecture.

The first is the split unit. Several studies keep only tumor-bearing slices or split at the slice level \cite{cheng2024egunetr,zhu2025bppnet,hu2024msml,qi2024addunet,ma2025msffe}. Adjacent slices of one patient can then fall into both training and test sets, and tumor-free slices, where false positives arise, are never scored. These studies report 82–90%. T-MPEDNet reports 89.1% after a random 80/10/10 split whose unit is not stated \cite{raghaw2025tmpednet}.

The second is aggregation. Global Dice pools all test voxels, so large lesions dominate. Per-case Dice averages one score per patient, so a missed small lesion weighs as much as a large one. PGC-Net reports both for the same model and test set: 85.72% global versus 73.63% per case \cite{you2024pgcnet}. This 12-point gap exceeds most claimed architectural gains. Many papers do not state their aggregation, a known pitfall \cite{reinke2024pitfalls}.

The third is the baseline. nnU-Net tumor Dice reported in other papers ranges from 66.57% \cite{bian2026mambaclinix} through 76.19% \cite{liu2025okmil} and 79.09% \cite{peng2024spidernet} to 89.54% \cite{ma2025msffe}. In the slice-level setting of MSFFE-Net, the proposed method exceeds nnU-Net by 0.07 points (89.61% vs. 89.54%) \cite{ma2025msffe}. Well-configured baselines are hard to beat under rigorous validation \cite{isensee2024revisited}. Table 3 therefore lists only methods retrained in our pipeline.

Studies that split by patient report lower values. MambaClinix reaches 71.37% (nnU-Net 66.57%) on a cohort mixing 161 in-house and 131 MSD Liver volumes \cite{bian2026mambaclinix}. PGC-Net reaches 73.63% per case, RIOUNeXt 76.83%, and weakly supervised OKMIL 65.22% \cite{you2024pgcnet,wang2025rioUNeXt,liu2025okmil}. The best per-case tumor Dice in the three LiTS challenge editions was 67.4%, 70.2% and 73.9% \cite{bilic2023lits}. A patient-level split alone does not guarantee comparability. Spider-Net (78/13/40 patients) and TCA-Net (101/10/20) report 85.94% and 83.90% without stating the aggregation \cite{peng2024spidernet,wu2025tcanet}, so we cannot place them. Our per-case tumor Dice of [[TBD: A6 per-case tumor Dice, 24 tumor-bearing test cases]] (global Dice [[TBD]]) should be read against the 65–77% band, not against the 85–90% values.

External validation needs similar care. LiTS and MSD Task03 contain the same scans \cite{simpson2019msd}. The LiTS training set also includes 20 scans from IRCAD and its test set none \cite{bilic2023lits}. 3D-IRCADb-01 likewise holds 20 IRCAD scans. A model trained on all LiTS training scans and tested on 3D-IRCADb therefore cannot be assumed to face unseen patients. The one cross-dataset result we found, EG-UNETR trained on LiTS and tested on 3D-IRCADb, fell from 84.45% (within-dataset, tumor slices only) to 45.40% \cite{cheng2024egunetr}. We therefore use [[TBD: TCIA Colorectal-Liver-Metastases or HCC-TACE-Seg]] as the external cohort.

## 5.2 What the mechanism analysis shows

The mechanism analysis (Section 4.8, Fig. 3) separates two properties that are easily conflated. Contrast invariance holds. In noise-free synthetic volumes, changing lesion contrast from 5 to 40 HU changed PC by at most $8.6\times10^{-3}$ per voxel, whereas gradient magnitude scaled linearly. This follows from the ratio form of PC \cite{kovesi1999phase,zhang2011fsim}. Reliability at low contrast does not follow from it. Kovesi's noise threshold $T$ is estimated from the finest-scale amplitude and has absolute units \cite{kovesi1999phase}. When contrast falls, the signal energy shrinks but $T$ does not. With $\sigma = 5$ HU white noise, $T$ suppressed PC almost everywhere at CNR $\le 1$. PC therefore abstains rather than guesses: it behaves as a high-specificity, low-sensitivity cue. This regime is common on LiTS: one third of training lesions have CNR below 1.28 (median 1.62). Two caveats apply. The analysis used the untrained operator on HU intensities, not the learned proxy features; and the threshold multiplier $k$ is learnable [[V2-pending: soft vs. hard threshold]]. The analysis thus predicts that any benefit should appear as false-positive suppression and boundary calibration, not as low-contrast recall. We stated this prediction before running the ablation study [[TBD: confirm timeline]].

**[[BRANCH A: ablation supports phase routing]]** Full PhASER-3D (A6) improved [[TBD: primary endpoint]] over the gradient-magnitude gate A1 by [[TBD: Δ, 95% CI]] and over the shuffled-PC control A7 by [[TBD: Δ, 95% CI]]. A7 matches A6 in parameters and compute, so the difference is attributable to the spatial content of PC rather than to added capacity [[CLAIM-PENDING: depends on A6 vs A0/A1/A7]]. As predicted, the gain concentrated in [[TBD: lesion-wise precision / FP per scan / HD95 / NSD]], while recall in the lowest CNR tertile changed by [[TBD]]. Phase evidence thus appears to help the network reject liver structures that are not lesion margins and to place the margins of visible lesions. It does not make invisible lesions visible. [[AUTHOR NOTE: if low-CNR recall also improves, state that the learned proxy and $k$ partly overcome the abstention seen with the fixed operator, and support this with Fig. 5.]]

**[[BRANCH B: ablation does not support phase routing]]** A6 did not differ from A1 [[TBD: Δ, 95% CI]] or from A7 [[TBD: Δ, 95% CI]]. The mechanism analysis is consistent with this null result. For many lesions the CNR lies near the regime where PC abstains, so the router receives an almost empty cue exactly where help is needed. At higher CNR, amplitude cues in the pretrained encoder probably locate the margin already. Contrast invariance is necessary but not sufficient; reliability under noise is the binding constraint. Any gain of A6 over A0 that A7 also achieves [[TBD]] reflects added capacity or regularization, not phase evidence. We report this negative result because the controls (gradient gate, shuffled PC, CNR strata) were designed to isolate a phase-specific effect.

## 5.3 Limitations

**Single split of one dataset.** All internal results come from fold 0 of one LiTS split, with 27 test cases (24 tumor-bearing). Confidence intervals are wide, and only key groups were repeated with several seeds [[TBD: seeds per group]].

**LiTS-only training.** The model was trained on contrast-enhanced CT from seven centers \cite{bilic2023lits}. Performance under other scanners, phases, or lesion types is tested only by one external cohort [[TBD]].

**Synthetic noise model.** The mechanism analysis used additive white Gaussian noise, the model under which Kovesi's threshold is derived \cite{kovesi1999phase}. Real CT noise is spatially correlated after reconstruction and resampling, and its variance depends on intensity. The direction of the finding follows from the absolute threshold; the CNR at which abstention begins does not, and must be re-measured under realistic noise.

**Isotropic filters on anisotropic voxels.** The log-Gabor bank is defined on a voxel-unit frequency grid, but voxels measure 1.0 mm along $z$ and 0.78 mm in-plane. A filter of a given wavelength in voxels therefore covers 1.28 times more physical distance along $z$. LiTS slice spacing also varies widely before resampling \cite{bilic2023lits}, so part of the $z$-structure seen by PC is interpolated.

**Prior inspection of the internal test set.** During early development, predictions of a preliminary model on the 27 internal test cases were inspected for error analysis. No V2 design choice or checkpoint was selected on these cases [[TBD: confirm]], but the set is not untouched. For this reason we treat the external cohort as the primary held-out evidence and the internal test set as secondary.

**Overlap with pretraining data.** The VoComni weights come from a pretraining collection that includes MSD tumor datasets \cite{wu2024largescale}, and MSD Task03 consists of the LiTS training cases \cite{simpson2019msd}. Images of our internal test cases may therefore have been seen during pretraining [[TBD: confirm from the VoComni case list]]. A0 and PhASER-3D share these weights, so their comparison is unaffected, but absolute internal scores may be optimistic. The from-scratch variant (A8) and the external cohort provide the uncontaminated evidence.

**Compute budget.** Each run used 200 epochs of 250 iterations on a single GPU [[TBD: GPU model; ablation budget if different]], one fifth of the nnU-Net default of 1000 epochs \cite{isensee2021nnunet}. All groups share this budget and report final-epoch weights, so comparisons are internally fair, but absolute values may be below those of fully converged models.

## 5.4 Future work

**Noise-calibrated reliability.** The hard threshold could be replaced by a graded significance score, such as local energy divided by a locally estimated, correlation-aware noise level. PC would then report weak evidence with low confidence instead of returning zero.

**Discriminative negatives.** The ranking loss contrasts the lesion margin only with the lesion interior and a peritumoral ring. PC responds equally to vessels and the liver capsule. Adding vessel-like regions and online false-positive components as hard negatives would teach the module which margins belong to lesions.

**Anisotropy-aware filters.** Defining the frequency grid in physical units, or using per-axis bandwidths, would make each scale correspond to the same physical size along every axis.

## Open questions for the authors

- Which external cohort is used (TCIA Colorectal-Liver-Metastases or HCC-TACE-Seg)? 5.1 and 5.3 depend on it.
- Will `reproduce_metrics.py` report global Dice alongside per-case Dice? 5.1 asks for both.
- The literature table (文献对比表.md §2, finding 5) describes the 66–77% band as "patient-level 3D" studies. PGC-Net, RIOUNeXt and OKMIL are 2D models, so I wrote "split by patient" instead. Please confirm the wording.
- Can you confirm the timeline claim in 5.2? Step 2 (2026-09-10) predicted false-positive suppression rather than low-contrast recall before any ablation was run. If true, it can be stated as a pre-specified prediction.
- Can you confirm that no V2 hyperparameter, loss weight, or checkpoint was chosen using the internal test cases (5.3)? The early error analysis on the 24 tumor-bearing cases (under- and over-segmentation counts, false-positive locations) used the Copy2 model.
- What is the final noise-threshold form (hard vs. soft) and the `PHASER_PC_MAX_VOXELS` setting for V2? Step 2 found that the 2^20 cap computes the shallowest PC at 64³. If V2 keeps the cap, add it to Limitations.
- What is the ablation budget? `validation_plan.txt` sets 100 epochs for ablations; the brief says 200 for all runs.
- How many seeds per group? This is needed in 5.3.
- Several BSPC volumes are printed as 2026 in the PDFs (MSFFE-Net 113, TCA-Net 112, BPP-Net 112, RIOUNeXt 112, OKMIL 115), but the bib keys say 2025. Please check `refs.bib` years.
