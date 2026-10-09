# 6. Conclusion

We presented PhASER-3D, a 3D liver-tumor segmentation network that adds phase-congruency-guided frequency routing to the skip connections of a VoComni-pretrained nnU-Net. Each routing module computes monogenic phase congruency with learnable log-Gabor scales. It uses this cue and a tumor prior to reweight low-, mid- and high-frequency Haar subbands. A zero-initialized residual gate makes each module an identity map at the start of training, which protects the pretrained features.

On a patient-level LiTS split, PhASER-3D reached a per-case tumor Dice of [[TBD]] on the 24 tumor-bearing test cases and [[TBD]] on the external cohort. [[CLAIM-PENDING: depends on A6 vs A0/A1/A7 — one sentence stating whether phase routing outperformed the gradient-magnitude gate and the shuffled-PC control, and on which metrics.]]

The mechanism analysis showed that phase congruency is contrast-invariant but abstains at low contrast-to-noise ratio, because Kovesi's noise threshold has absolute units. Phase evidence is therefore better suited to suppressing false positives and calibrating boundaries than to recovering low-contrast lesions. [[CLAIM-PENDING: keep "consistent with the ablation results" only under Branch A.]]

Finally, reported LiTS tumor Dice depends strongly on the split unit and the aggregation. Results should be compared only under matched, patient-level, per-case protocols. Code, split files, and evaluation scripts are released [[TBD: repository URL]].

## Open questions for the authors

- Should the conclusion name the external cohort explicitly, or refer to it only as "an independent external cohort"?
- If Branch B holds, should the second paragraph instead state the null result in one sentence ("phase routing did not outperform the amplitude and shuffled-PC controls")? The current wording is neutral on purpose.
- Will the code and split files be public at submission (BSPC encourages data and code availability statements)? If not, change the last sentence to "available on request".
