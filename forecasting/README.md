# Hierarchical machine-specific dynamics

## Reproduce forecasts

Python 3.11+ and numpy==2.3.5. Organizers kindly install requirements.txt before offline execution, per the supplied 60/40 kit rules. No network or external API is used during inference.

python predict.py --data /absolute/path/to/public --out /absolute/path/to/submission.csv

The ZIP must be extracted with README.md, reproduce.json, predict.py, dynamics.py, config.json, model_0.npz, and requirements.txt at its root. Saved training priors and frozen settings are loaded relative to predict.py. Input files are only test/queries.csv and test/history.csv; test labels, conditions, targets, reference forecasts, and uploaded forecast CSV are not read. Output is written only to the supplied --out path.

## Model built from scratch

The likelihood factorizes into P(win) and P(payout class | win). Both heads maintain independent machine-specific posteriors over 151 candidate structures: a constant model; single lags 1-8 and all lag pairs using binary win/loss, large-payout, and coarse payout contexts; and peer/cross-machine lag contexts. Positive payouts retain all four classes, so the output remains probabilities for 0,1,2,5,20.

Conditional Dirichlet counts adapt to each machine's observed history. Bayesian structure weights use prequential log predictive evidence: each historical prediction is scored before its outcome updates the counts. Shared structure priors are estimated per machine type with a single strongly regularized empirical-Bayes update (64 pseudo-machines). Family-balanced initial priors avoid rewarding a feature family simply because it has more candidate structures.

The selected model has count half-life 12 ticks, structure-evidence half-life 96, Dirichlet strength 2, and posterior temperature .5. Thus recent odds adapt faster than structural beliefs. Missing tick contexts back off to local recency estimates; missing ticks are never treated as losses. Peer features use strictly earlier ticks. Exact-lag contexts are vectorized, with causal prefix-equivalence tests.

## Training and overfitting controls

Only main training histories and observed training outcomes were used to fit shared priors. No validation true probabilities were used as training labels. Training episodes were stratified by condition into 48 fit, 24 selection, and 24 audit episodes. Selection evaluated observed-outcome Brier loss at four causal cutoffs per held-out machine, averaging by episode and condition. Episodes are never split across fit/selection/audit for these comparisons. Local inference on an evaluation machine uses only its permitted historical observations.

Stage one compared four fixed configurations; a training-motivated memory refinement compared four new configurations, the stage-one winner, and one fixed adaptive-memory mixture. The final configuration was selected on TRAIN selection episodes, then priors were refitted on all 96 training episodes. Validation scores were inspected only after each stage was frozen. Validation had already been repeatedly inspected for earlier models; training audit was also reused after the first hierarchical stage. Neither is claimed to be a newly untouched final holdout. Zero overfitting cannot be guaranteed.

Query signal is not assigned to historical pseudo-examples, and this model does not use it. Peer machines are indexed by sorted machine_id within the episode; this supplies local relative offsets, not cross-split identity features. Conditions are used only for training split balance and evaluation, never inference.

Training is deterministic; no stochastic optimizer or random initialization is used. Split ordering uses SHA256 with prefix hierarchical-v1:. Diagnostic bootstrap uses seed 2718; bootstrap results do not affect the model.

## Refit frozen settings

python train_frozen.py --data /absolute/path/to/public --out /new/model/directory --config config.json

This reproduces the saved prior fitting using only train/ data and the already chosen configuration. It writes model_0.npz and config.json to a new directory. predict.py and dynamics.py can then be placed alongside those files. Changing the data requires fresh validation and predictions, especially for adaptation.

## Team

Outliers

## Verification and limits

See training_report.json and RESULTS.md for measured scores, runtime, file limits, and reproduction results. The format checker does not reveal hidden test score. The model has not been certified on the organizer's Linux/Python environment; local reproduction uses the pinned NumPy package. No claims are made that this model reaches competitors' reported scores or guarantees a prize. Existing earlier submissions remain preserved separately.
