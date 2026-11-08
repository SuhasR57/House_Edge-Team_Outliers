# Hierarchical dynamics model — frozen result

## Recommendation

Use this hierarchical model in place of the earlier temporal-only entry, subject to the competition's normal reproduction checks. Full validation: 47.458381/60. This is substantially stronger than the earlier 31.366494/60 but remains below the competitors' reported 50-52 main-round points. No hidden-test score or prize outcome is known.

## Results

| Model | Earlier tuning half | Earlier audit half | Full validation |
|---|---:|---:|---:|
| Earlier temporal-only | 31.399547 | 31.320900 | 31.366494 |
| Initial hierarchical dynamics | 41.016062 | 40.040115 | 40.547716 |
| Frozen fast-count / slow-structure hierarchy | 47.414590 | 47.513065 | 47.458381 |

The new hyperparameters were selected on episode-disjoint TRAIN data, not by choosing the highest score in this validation table. Both old validation halves were repeatedly inspected in earlier work, so neither is presented as an untouched holdout.

| Condition | Earlier temporal loss ratio | New hierarchical loss ratio |
|---|---:|---:|
| Stable | .938393 | .239399 |
| Short history | .987399 | .370292 |
| Drift | .898661 | .222812 |
| Shared change | .827024 | .224560 |

Ratios are errors divided by the organizer's frozen pooled reference, following official episode/condition aggregation. Lower is better. The new mean ratio is .264266, compared with approximately .912869 for the earlier temporal model. Short history remains the weakest condition.

## What changed

Custom NumPy implementation, independent of the starter forecaster at inference. It factorizes win probability and positive payout size, with 151 candidate lag structures for each head. Machine-local prequential evidence learns which structures work for that machine; type-level structure priors are shared across training machines and strongly regularized. The selected model uses 12-tick conditional-count memory and 96-tick structure-evidence memory, shrinkage strength 2, and posterior temperature .5. Local odds can change quickly without forgetting all evidence about the machine's temporal relationships.

## Overfitting controls and limitations

- Train split: 48 fitting episodes, 24 selection episodes, 24 audit episodes, stratified by the four conditions. Entire episodes stay together.
- Training selection uses observed outcomes at four causal historical/next-spin anchors, with equal episode/condition aggregation. No validation true probabilities are used to fit priors or optimize parameters.
- Initial study: four fixed configurations. Follow-up: four fixed fast-memory variants, initial winner, and one fixed adaptive-memory candidate. Selected by training-selection Brier only.
- Selection Brier: .438901 versus .562848 for a type-prior recency reference at identical anchors. Reserved-train audit: .471658 versus .588164. These are observed-outcome Brier losses, not competition points.
- Shared structure distributions are shrunk toward family-balanced priors with 64 pseudo-machines; per-context outcome counts are also regularized.
- Audit was reused after the first hierarchical experiment, and validation informed broad architectural diagnosis after that stage. This is not proof of zero overfitting. Settings are now frozen; no additional validation-driven tuning was performed after the 47.46 result.
- A 2,000-resample condition-stratified paired episode bootstrap gives a descriptive 95% interval of [15.33,16.86] points for improvement over temporal-only. Candidate score interval: [46.58,48.30]. These intervals do not correct for model-development bias or guarantee hidden-test performance.

## Causality and correctness checks

Passed exact-lag vectorization checks, normalized probability checks, missing-history fallback, and rejection of current/future history rows. Altering all outcomes at/after a historical prediction tick leaves that forecast unchanged. Replaying only the history prefix gives exactly the same prediction as requesting that anchor from the full episode. Other-machine features also use strict earlier-tick lags.

## Submission format

Upload one forecast CSV under Forecast CSV (10,000,000-byte limit) and one code ZIP under Code ZIP (100,000,000-byte limit). The ZIP has README.md and reproduce.json directly at its root. numpy==2.3.5 is pinned in root requirements.txt; organizer dependency preparation is allowed by the supplied 60/40 kit. Inference is offline and uses saved priors plus permitted test histories.

The package includes predict.py, dynamics.py, frozen config/model, training refit command, tests, and training report. Earlier temporal-only CSV and ZIP are preserved. No submission has been uploaded.
