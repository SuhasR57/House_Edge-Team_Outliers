# EdgeShift Lab

### Hierarchical forecasting with evidence-grounded Gemini research review

**Team Outliers · TAMU Datathon 2026 · House Edge Challenge**

EdgeShift Lab forecasts payout probabilities for machines whose behavior can change over time. It combines a hierarchical Bayesian dynamics model, implemented from scratch in NumPy, with a Gemini-powered reviewer that turns experiment results into an inspectable research report.

The project addresses two connected problems: learning useful temporal patterns from limited observations, and explaining which experiments actually improved the model. The forecasting engine produces reproducible offline predictions; Gemini helps the team interpret measured results, identify weaknesses, and propose bounded follow-up experiments.

**Measured outcome:** the frozen hierarchical model achieved **47.46 / 60 on public main-round validation**, compared with **31.37 / 60** for our earlier temporal baseline. These are validation results.

## 1. Challenge and objective

For each query, the model predicts a probability distribution over five payout classes:

```text
P(payout = 0), P(payout = 1), P(payout = 2), P(payout = 5), P(payout = 20)
```

The data includes stable machines, short histories, distribution drift, and changes shared across machines within an episode. A useful model must balance rapid adaptation with enough pooling to avoid reacting too strongly to sparse observations.

The main-round score is calculated as:

```text
60 / (1 + mean normalized condition error)
```

Higher points and lower normalized error are better. Points are not classification accuracy. The results in this README describe the main round only.

## 2. System overview

```text
Observed training histories
    -> shared payout and structure priors
    -> frozen model configuration and checkpoint

Permitted test histories + queries
    -> causal machine-specific posterior updates
    -> five-class probability forecasts
    -> submission.csv

Recorded experiment metrics
    -> allowlisted evidence packet
    -> Gemini structured research review
    -> local schema and reference validation
    -> HTML report + JSON review + API receipt
```

The two components have distinct responsibilities. The numerical model computes competition forecasts. Gemini explains experiment evidence and suggests research directions; it does not generate or modify the submitted probabilities.

## 3. Model approach

### Factorized payout prediction

The model separates the probability of any positive payout from the distribution of payout size:

```text
P(Y = 0) = 1 - P(win)
P(Y = k) = P(win) × P(Y = k | win), for k in {1, 2, 5, 20}
```

This lets the model learn whether a machine pays out and how large its positive payouts are using separate posterior estimates, while retaining the complete five-class output distribution.

### Machine-specific temporal structures

Each prediction head maintains posterior weights over **151 candidate structures**. These include a constant model, individual lags from 1 to 8, lag-pair contexts, binary win/loss patterns, large-payout indicators, coarse payout contexts, and lagged peer-machine contexts.

Structure weights are updated using **prequential predictive evidence**: an observation is scored using the prediction made before that observation updates the model. This supports causal adaptation instead of evaluating a pattern on outcomes it has already incorporated.

### Hierarchical pooling and sparse histories

Shared priors are learned by machine type. Machine-specific conditional Dirichlet counts then adapt those priors to each observed history. Structure priors use a strongly regularized empirical-Bayes update with **64 pseudo-machines**, and family-balanced initialization prevents larger feature families from receiving more prior mass merely because they contain more candidates.

When an exact lag context is unavailable, inference backs off to local recency estimates. Missing ticks are not interpreted as losses.

### Separate memory for odds and structure

The selected configuration, `fast12_structure96`, uses:

| Parameter | Frozen value | Purpose |
| --- | ---: | --- |
| Count half-life | 12 ticks | Adapt payout estimates to recent observations |
| Structure-evidence half-life | 96 ticks | Retain temporal-pattern evidence longer |
| Dirichlet strength | 2 | Regularize conditional probabilities |
| Posterior temperature | 0.5 | Control concentration of structure weights |

Peer-machine contexts use strictly earlier ticks within the same episode. Condition labels are used for split balancing and evaluation, not inference. The current model does not use the query signal.

## 4. Evaluation and overfitting controls

The 96 training episodes were divided into **48 fit, 24 selection, and 24 audit episodes**, stratified by condition. An episode stays entirely within one split. Candidate selection uses observed-outcome Brier loss at four causal cutoffs per held-out machine, aggregated by episode and condition.

Development used small, fixed configuration comparisons. The memory-refinement stage compared four new settings, the earlier selected configuration, and one adaptive-memory mixture. The winning configuration was selected on training-selection episodes; shared priors were then refitted on all 96 training episodes.

Validation true probabilities were not used to fit model parameters. However, public validation was inspected during earlier development, and the training audit was reused after the first hierarchical stage. These are **not untouched final holdouts**, and the project does not claim zero overfitting.

For future experiments, compare candidate and baseline on the **same episodes and cutoffs**. In particular, the training-audit Brier score must not be used as a threshold for a different selection split. Preserve the frozen model as the fallback unless a controlled comparison supports a change.

## 5. Results

### Public main-round validation

| Metric | Temporal baseline | Hierarchical model |
| --- | ---: | ---: |
| Points out of 60 | 31.3665 | **47.4584** |
| Mean normalized condition error | 0.9129 | **0.2643** |

The measured improvement is **16.09 validation points**.

| Condition | Temporal normalized error | Hierarchical normalized error |
| --- | ---: | ---: |
| Stable | 0.9384 | **0.2394** |
| Short history | 0.9874 | **0.3703** |
| Drift | 0.8987 | **0.2228** |
| Shared change | 0.8270 | **0.2246** |

The hierarchical model improved all four reported conditions. **Short history remains its weakest condition**, making sparse-history pooling a useful direction for further controlled experiments.

### Selection, uncertainty, and reproduction

| Measurement | Recorded result |
| --- | ---: |
| Selected configuration: training-selection Brier | 0.438901 |
| Reused training-audit Brier | 0.471658 |
| Descriptive bootstrap interval for validation points | 46.58–48.30 |
| Descriptive bootstrap interval for point gain | 15.33–16.86 |
| Forecast queries in recorded reproduction | 1,536 |
| Maximum probability difference in reproduction | 0.0 |
| Forecast CSV size | 202,347 bytes |

The bootstrap uses 2,000 episode resamples with seed 2718. It describes sampling uncertainty on inspected validation data; it does not correct for model-selection bias or predict hidden-test ranking.

The recorded local reproduction produced a byte-identical forecast CSV. Its historical archive hash applies only to the original verified archive; repackaged ZIP files require their own verification. Runtime varies by environment, and organizer-environment certification has not been established.

Detailed model results are included in `forecasting/training_report.json`, `forecasting/RESULTS.md`, and `forecasting/uncertainty.json`.

## 6. Gemini API integration

Gemini serves as an **experiment-review assistant**. It receives aggregate results and a research question, then returns a structured review containing a summary, findings, proposed experiments, limitations, and evidence references.

The client uses the Google Gemini Interactions API with `gemini-3.8-flash`, API revision `2026-05-20`, and `store=false`. The local application:

- Builds an allowlisted evidence packet from recorded experiments.
- Requests a JSON response with a defined output structure.
- Rejects unknown or missing evidence references and malformed review fields.
- Renders numeric result cards directly from recorded evidence.
- Escapes generated text before inserting it into HTML.
- Saves a receipt with model information, timing, available token usage, and content hashes.

**Live integration has been demonstrated.** A saved successful review is included in `live-review/report.html`, with its receipt in `live-review/receipt.json`. New runs create separate timestamped directories under `reports/`.

The client allows 180 seconds per attempt and one retry after five seconds for temporary connection failures or HTTP 500/502/503/504. Authentication, schema, and quota errors do not automatically retry. A timed-out request may still incur provider usage.

### Data handling and interpretation limits

API keys are loaded from a private local `.env` file or process environment and sent in the authentication header. The project ZIP excludes `.env`. The review payload excludes raw histories, individual test forecasts, checkpoints, credentials, and absolute local paths.

Reference validation checks structure and evidence IDs; it does not prove that generated interpretations are correct. For example, an experimental proposal must be reviewed to ensure it compares scores from the same evaluation split. Generated recommendations remain hypotheses until tested.

The Gemini integration has **not independently increased the measured forecasting score**. Its contribution is evidence review, communication, and experiment planning.

## 7. Setup and usage

### Requirements

- Python 3.11 or newer.
- `numpy==2.3.5` for forecasting.
- A Gemini API key and network access for a new live review.
- The organizer's participant data for forecast reproduction; data is not bundled.

The reviewer uses only Python's standard library. Saved HTML reports can be opened without an API key.

### Reproduce forecasts

From the project root:

```powershell
python -m pip install -r forecasting/requirements.txt
python forecasting/predict.py --data "C:/path/to/public" --out "C:/path/to/submission.csv"
```

Set `--data` to the directory containing the organizer's `test/` folder. Inference loads the frozen configuration and checkpoint beside `predict.py` and reads permitted test histories and queries. It runs without network access or Gemini credentials.

To refit shared priors using the frozen settings:

```powershell
python forecasting/train_frozen.py --data "C:/path/to/public" --out "C:/path/to/new-model" --config forecasting/config.json
```

This refits the chosen configuration; it does not rerun the original configuration search. See `forecasting/README.md` for model-file placement and reproduction details.

### Run a Gemini review

Create a key in [Google AI Studio](https://aistudio.google.com/apikey). Copy `.env.example` to `.env` if a private `.env` does not already exist, then set:

```dotenv
GEMINI_API_KEY=your_key_here
```

Keep this file local. From PowerShell in the project folder, run:

```powershell
.\RUN_LIVE_REVIEW.cmd
```

The Windows launcher uses the bundled Python runtime when available, otherwise `py -3`. Alternatively:

```powershell
python edge_review.py
python edge_review.py --question "Which condition is weakest, and what controlled experiment should we test next?"
```

Open the printed `report.html` path after success. Process environment credentials take precedence over `.env`. For an evidence-only preview with a new output directory:

```powershell
python edge_review.py --preview --out preview-new
```

### Run checks

```powershell
python test_edge_review.py
python forecasting/test_dynamics.py
```

The reviewer has **27 passing offline contract tests**, including reference validation, credential handling, HTML escaping, bounded retries, and failure behavior. Provider responses in these tests are explicit fixtures; the separate live receipt documents an actual API call. Forecasting tests include causal prefix-equivalence checks.

## 8. Project layout

```text
edge-shift-gemini-project/
├── README.md                 Project overview and usage
├── edge_review.py            Gemini client, validation, and report rendering
├── build_evidence.py         Aggregate evidence extraction
├── test_edge_review.py       Reviewer contract tests
├── RUN_LIVE_REVIEW.cmd       Windows launcher
├── .env.example              Credential template; no real key
├── evidence.json             Recorded aggregate evidence
├── integration_status.json  Integration status snapshot
├── SPONSOR_DEMO.md           Demo and sponsor-submission notes
├── live-review/              Saved successful review and receipt
├── preview/                  Evidence-only report
└── forecasting/              Offline model, checkpoint, tests, and results
```

A local `.env` and new `reports/` directories are created during use. Neither is required to reproduce offline forecasts.

## 9. Submission and next steps

Use the dedicated forecast CSV and offline reproduction ZIP for the House Edge forecasting submission. The combined project bundle supports the broader project demonstration; it is not a replacement for the competition's required offline package layout.

Gemini is the integrated service for a potential **Best Use of Gemini API** sponsor entry. Category registration, eligibility, and awards are determined by the organizers; a successful API call does not itself submit or qualify the project.

Further work should focus on stronger pooling for sparse histories, small training-only memory comparisons, and an independent evaluation before adopting additional changes. The frozen configuration remains the reproducible baseline for those comparisons.

## Team and acknowledgments

Built by **Team Outliers** for the TAMU Datathon 2026 House Edge Challenge. The organizer's participant kit supplies the forecasting task, data, and scoring protocol. Google Gemini supplies the research-review service.
