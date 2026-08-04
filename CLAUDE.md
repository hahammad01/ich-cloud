# Project context for Claude Code

This is a from-scratch benchmark of CNN backbones for **multi-label** intracranial
hemorrhage (ICH) subtype classification on the RSNA 2019 Brain CT dataset,
intended to run on Google Colab or Kaggle Notebooks.

When reviewing or amending this code, treat the following as **non-negotiable
invariants**. Flag any change that would violate them.

1. **Patient-level splitting is sacred.** Train/val/test are split by `patient_id`
   (see `prepare_data.py`). Never switch to a random per-slice split — it leaks
   correlated slices across splits and inflates every metric. The assertion in
   `make_split` guards this; do not remove it.

2. **The task is multi-label, not multi-class.** 6 independent sigmoid outputs
   (`epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any`),
   trained with `BCEWithLogitsLoss`. A single slice can be positive for several
   subtypes. Never replace this with softmax / cross-entropy.

3. **Fair comparison = identical training config.** Both backbones must use the
   same split, image size, epochs, schedule, augmentation, and seed (all in
   `config.py`). If you add capacity to one model, note the confound explicitly.

4. **EDH is the rare, high-stakes class.** Keep per-subtype metrics visible.
   Report PR-AUC and PPV alongside ROC-AUC (`evaluate.py`) — a good AUC can hide
   a terrible false-alarm rate on EDH. Do not average EDH away into a macro number
   without also showing it separately.

5. **Keep inferences labelled as inferences.** If you note a suspected cause
   (e.g. "over-prediction likely from a single global threshold"), keep it phrased
   as a hypothesis until there is evidence, not an established finding.

## Suggested tasks to ask Claude Code
- Sanity-check the DICOM HU / windowing math in `dataset.py`.
- Verify no patient leakage after `prepare_data.py` runs (re-derive the assertion).
- Add per-subtype threshold tuning to `evaluate.py` (currently one global threshold).
- Add a focal-loss option as an alternative to BCE pos_weight.
- Add optional 2.5D input (stack neighbouring slices) — biggest expected EDH gain.
