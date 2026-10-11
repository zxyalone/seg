# A9 / A11 / A12 pilot execution

`server_lowcnr_pilots.py` runs A9_s0 on GPU 0, A11_s0 on GPU 1 and
A12_s0 on GPU 2 from a committed, isolated worktree. A10 is a gradient
preflight control only. All runs use 200 epochs, fold 0, and the same
VoComni checkpoint used by A0_s0/A1_s0. The test set is never inferred.

Live logs and atomic state are stored in:
`/root/autodl-tmp/nnUNet_data/nnUNet_results_v2/_pilot_lowcnr/`.
The controller records first epoch times and estimated training duration.
Creating `STOP` in that directory stops its jobs and prevents publication.
A newly written global stop marker also stops this controller. Stop markers
predating this separately authorized pilot are retained; the old E11/E12
queues are not resumed.

After every run exits successfully, the controller verifies 200 completed
epochs, the initialization path, 21 validation masks and eval_val files.
`lowcnr_pilot_metrics.py` removes components with equivalent spherical
diameter <4 mm from the saved masks and reruns the existing Hungarian
matching. Its unfiltered detection/FP counts must reproduce eval_val.
The pre-registered gates are unchanged: gain >=5 percentage points in
low-CNR recall and FP/scan increase <=0.3 relative to A1_s0. All three
pilots are reported; passing does not automatically start cross-validation.

With `--publish`, raw eval_val, manifests, logs, preflight evidence and
filtered metrics are committed and pushed to main from the isolated
worktree. Remote changes are merged without force. Failures, stops and
merge conflicts remain in the live state and do not publish incomplete
results as complete. No checkpoints or full-resolution volumes are added
to Git. Unrelated edits in the original worktree are not staged.
