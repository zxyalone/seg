# Supplementary Material

## S1 Protocols of recent LiTS studies

**Table S1.** Protocols and reported tumor Dice of recent LiTS studies. Values are copied from each paper as reported (%). The comparability column assesses each protocol against ours: patient-level split, whole 3D volumes, per-case Dice averaged over tumor-bearing test cases. Studies are ordered by comparability. A dash means not reported.

| Study | Venue/year | Dimensionality | Split unit and sizes (LiTS) | Test samples | Dice aggregation | Reported tumor Dice | nnU-Net in same paper | Comparable to ours? |
|---|---|---|---|---|---|---:|---:|---|
| LiTS challenge \cite{bilic2023lits} | MedIA 84 (2023) | Any | Patient; official 131 train / 70 test | 70 volumes (labels withheld) | Per case | 67.4 / 70.2 / 73.9ᵃ | — | Reference |
| MambaClinix \cite{bian2026mambaclinix} | BSPC 124 (2026) | 3D (nnU-Net framework) | Patient; 209 / 53 / 30ᵇ | 30 volumes | Not stated | 71.37 | 66.57 | Yes, with caveatᵇ |
| PGC-Net \cite{you2024pgcnet} | BSPC 92 (2024) | 2D (352 × 352) | Patient; 20 random volumes for test; five-fold CV | 20 volumes | Per case and global, both reportedᶜ | 73.63 per case / 85.72 global | — | Yes (per-case value) |
| OKMIL \cite{liu2025okmil} | BSPC 115 (2026) | 2D | Patient; 79 / 30 (109 tumor-bearing volumes) | 30 volumes | Not stated | 65.22 (weakly supervised) | 76.19 (fully supervised)ᵈ | Partly: different task (weak supervision) |
| RIOUNeXt \cite{wang2025rioUNeXt} | BSPC 112 (2026) | 2D (448 × 448) | Patient, sequential; volumes 27–130 train/val (3:1), 0–26 test | 27 volumes | Not stated | 76.83 (five-fold CV 76.91) | — | Partly: aggregation unknown |
| Spider-Net \cite{peng2024spidernet} | BSPC 93 (2024) | 2D (256 × 256) | Patient; 78 / 13 / 40 | 40 volumes | "Average Dice", unit not stated | 85.94 | 79.09 (listed as 3D CNN) | Partly: aggregation unknown |
| TCA-Net \cite{wu2025tcanet} | BSPC 112 (2026) | 2.5D (3 adjacent slices, 256 × 256 crops) | Patient; 101 / 10 / 20 | 20 volumes (2107 slices) | Not stated | 83.90 | — | Partly: aggregation unknown |
| EG-UNETR \cite{cheng2024egunetr} | BSPC 97 (2024) | 2D | Tumor-bearing slices only; five-fold CV | Tumor-bearing slices | Slice-wise, aggregation not stated | 84.45ᵉ | — | No |
| BPP-Net \cite{zhu2025bppnet} | BSPC 112 (2026) | 2D | Tumor-bearing slices only; random 7:1:2 | Tumor-bearing slices | Slice-wise, aggregation not stated | 82.06 | — | No |
| MSML-AttUNet \cite{hu2024msml} | BSPC 99 (2025) | 2D (256 × 256) | Slice; 5752 / 1438 of 7190 slices | 1438 slices | Slice-wise, aggregation not stated | 87.74 | — | No |
| AD-DUNet \cite{qi2024addunet} | BSPC 95 (2024) | 2D (512 × 512) | Slice; random 8:1:1 | Slices | From TP/FP/FN, unit not stated | 89.45 | — | No |
| T-MPEDNet \cite{raghaw2025tmpednet} | BSPC (2025), 108225 | 2D (256 × 256) | Random 80 / 10 / 10, unit not statedᶠ | Not stated | Not stated | 89.1 | — | No: protocol cannot be matched |
| MSFFE-Net \cite{ma2025msffe} | BSPC 113 (2026) | 2D (256 × 256) | Tumor-bearing slices only; slice-level 8:1:1 (3987 / 498 / 499) | 499 slices | Slice-wise | 89.61 | 89.54 | No |

Notes. CV, cross-validation; MedIA, *Medical Image Analysis*; BSPC, *Biomedical Signal Processing and Control*. Volume and year are as printed on each article. ABeDice \cite{zheng2025abedice} is not listed because it does not report a LiTS tumor task.

ᵃ Best per-case tumor Dice of the ISBI 2017, MICCAI 2017 and MICCAI 2018 editions on the hidden test set.
ᵇ The LiverT cohort mixes 161 in-house contrast-enhanced CT volumes with the 131 MSD Liver (= LiTS training) volumes. The split is shuffled across both sources.
ᶜ PGC-Net defines Dice global (DG) over all test volumes pooled, and DSC as the mean of per-case Dice. Values are means over five-fold cross-validation.
ᵈ OKMIL's comparison table states that some baseline metrics are cited from the original articles.
ᵉ Trained on LiTS and tested directly on 3D-IRCADb, EG-UNETR reaches 45.40. The LiTS training set contains 20 IRCAD scans \cite{bilic2023lits}, so even this transfer is not fully independent.
ᶠ The paper slices volumes into 2D images before splitting "CT images" 80/10/10; whether the unit is the patient or the slice is not stated.

## S2 Reproducibility checklist

Items marked [[TBD]] are not yet fixed or not yet verified for the V2 runs.

| Item | Content | Status |
|---|---|---|
| Code | Fork of nnU-Net v2 \cite{isensee2021nnunet}: trainer `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_phaser.py` and modules `phaser/{unet,modules,losses,dim_utils}.py`. Repository URL [[TBD]]. Upstream nnU-Net version or commit [[TBD]]. License [[TBD]]. | [[TBD]] |
| Environment | Python 3.10, PyTorch 2.1.0 + CUDA 11.8, SimpleITK 2.5.6, single NVIDIA RTX 3090 (24 GB) for earlier runs. Confirmation for V2 runs and a full package lock file [[TBD]]. | [[TBD]] |
| Pretrained weights | VoComni encoder weights \cite{wu2024largescale}, file `VoComni_nnunet.pt`; 282 of 292 keys loaded, 10 segmentation-head keys re-initialized. Download source [[TBD]]; SHA-256 [[TBD]]. | [[TBD]] |
| Per-run manifest | One JSON file per run recording: git commit of the code; all `PHASER_*` and `nnUNet_*` environment variables; random seed; epochs and iterations actually completed; SHA-256 of the final checkpoint; GPU model; start and end time; path of the training log. Seed control (`PHASER_SEED`) implemented; written automatically to `run_manifest.json` in the run folder. | Implemented (`nnUNetTrainer_phaser.py`) |
| Data and splits | LiTS training set (131 volumes) as nnU-Net `Dataset102_LiTS`, `3d_fullres`, `nnUNetPlans`, target spacing 1.0 × 0.78 × 0.78 mm. Fold 0 `splits_final.json` (83 train / 21 validation) and the list of 27 test case IDs, including the three tumor-free cases (LiTS_032, LiTS_106, LiTS_115). External-cohort case IDs and preprocessing [[TBD]]. | Split files to be released [[TBD]] |
| Model selection | All model, hyperparameter and post-processing choices made on the 21 validation cases. Internal test cases evaluated once at the end (see Section 5.3 on earlier error analysis). | Stated |
| Training schedule | 200 epochs × 250 iterations for every group, SGD (Nesterov, momentum 0.99), poly learning-rate decay, AMP. Per-group switches in `experiments/configs/<group>.env`; queue in `experiments/queue.sh`. | Released with the code |
| Checkpoint | Final-epoch weights reported for every run; no checkpoint selection on validation pseudo-Dice or on test data. | Stated |
| Inference | nnU-Net sliding-window inference, no test-time augmentation (`--disable_tta`). Sliding-window step size 0.5, Gaussian weighting; also no TTA in the end-of-training validation. Post-processing [[TBD: none, or rule chosen on validation]]. | [[TBD]] |
| Evaluation scripts | `reproduce_metrics.py`: per-case Dice and IoU for liver parenchyma (label 1), whole liver (label > 0) and tumor (label 2); 24-case and 27-case cohorts. `experiments/eval/evaluate.py`: the same per-case overlaps plus global Dice (pooled voxel counts); lesion-wise recall, precision, F1 and FP per scan (26-connected components, Hungarian matching at IoU > 0.1 and > 0.5); tumor HD95 and NSD (τ = 2 mm). `experiments/eval/test_eval.py` checks all of these on synthetic volumes with known answers. | Available |
| Tumor-free cases | In the 27-case cohort, a tumor-free case with any predicted tumor voxel scores tumor Dice 0. An empty prediction on an empty reference scores 1. | Fixed in `evaluate.py` |
| CNR strata | Lesion–liver CNR with a liver ring 2–6 voxels outside the lesion; tertile thresholds 1.2785 / 1.9208 frozen on the 518 training lesions. Training-lesion thresholds: `step2_mechanism/scripts/s42_real_pc.py`; test lesions: `experiments/eval/evaluate.py`, same rule (resampling to the target spacing, 6-connected lesions ≥ 50 voxels for the thresholds, ring at voxel distance 2–6). | Available |
| Statistics | Paired bootstrap 95% CI and Wilcoxon signed-rank test on per-case values; 10,000 bootstrap resamples over cases; rank-biserial effect size; Holm correction (`experiments/eval/stats.py`). Seeds 0, 1, 2 for A0 and A6; seed 0 for other groups. | Available |
| Mechanism analysis | Synthetic and real-case PC analysis scripts (`step2_mechanism/scripts/`). Route selectivity in `s44_routes.py` corrected (per-band absolute differences) and checked on synthetic routes; rank correlation in `s44b_pc_cap.py` corrected (average ranks for ties). Rerun with the V2 operator and trained V2 models [[TBD]]. | Partly available |

## S3 Gradient check of the V2 training code

Before any V2 run, `experiments/check_grads.py` trains every ablation group for 20 steps on synthetic CT-like volumes (64³, lesion CNR 0.3–1.7), using the unmodified training step and optimizer configuration. It records, for every PhASER parameter tensor, whether it ever receives a non-zero gradient and its largest gradient magnitude. No group has a parameter without gradient. In A6, A7 and A8 the proxy and log-Gabor parameters of all five levels receive gradients of order 10⁻⁴–10⁻³ through $\mathcal{L}_{\mathrm{rank}}$. In the groups without $\mathcal{L}_{\mathrm{rank}}$ (A3–A5) the same parameters receive gradients below 10⁻⁷, because their only path is scaled by $\tanh(\gamma_l) \approx 0$. The first router convolution receives small gradients in all PFCR groups until its zero-initialized output convolution moves (Section 3.3). [[TBD: rerun at 128³ for 50 steps on the GPU and report the log]]

## Open questions for the authors

- T-MPEDNet is marked "No" because its split unit is not stated. 文献对比表.md marks it as not comparable without this nuance. Should it read "Unclear" instead?
- MSML-AttUNet's 7190 slices equal the number of tumor-bearing slices that BPP-Net reports for LiTS. MSML may therefore also use tumor-bearing slices only, but the paper does not say so, and I did not claim it. Keep as is?
- 文献对比表.md says Spider-Net retrained all models under 2D preprocessing. I could not confirm this in the paper text; the paper lists nnU-Net as a 3D CNN. I omitted the claim. Please check.
- 文献对比表.md describes RIOUNeXt as a "2D cascade". I found only "2D, 448 × 448" in the text and wrote that.
- Several venue years differ from the bib keys: MSFFE-Net, TCA-Net, BPP-Net, RIOUNeXt and OKMIL are printed as 2026, and MSML-AttUNet as 99 (2025). Please align `refs.bib`.
- Is the VoComni md5 recorded in the repository README (`ef44f2fb…`) the file used for V2? Should S2 report SHA-256 only?
- Will the 27-case cohort follow the LiTS convention (empty prediction on empty reference = Dice 1)? `reproduce_metrics.py` currently scores 0 when there are false positives and does not document the empty–empty case.
- Is post-processing used (e.g., largest connected component for liver)? The brief mentions post-processing choices but does not define any.
