# FinML Platform

Production-grade probability-of-default modelling with leakage-aware validation,
calibrated probabilities, explainability and CCAR-style stress testing. Built to
the expectations of SR 11-7 and the ECB guide to internal models: every score is
reproducible, explainable and traceable to the model version that produced it.

The platform runs **fully offline** out of the box — no API keys, no network, no
GPU. A synthetic panel with genuine macro factor structure stands in for real
data, so `make train` works on a cold laptop and in CI identically.

---

## Quick start

```bash
make install
```

```bash
make train-fast
```

```bash
make serve
```

```bash
make dashboard
```

`train-fast` completes in about 20 seconds and writes a promoted champion to
`artifacts/`. The API loads it at startup; the dashboard consumes the API.

Full run with hyperparameter tuning and the model-risk evidence pack:

```bash
make train
```

---

## Architecture

Clean Architecture with the dependency rule enforced mechanically — a unit test
fails the build if it is violated.

```
domain/          Pure business logic. Standard library only.
   ▲                PD arithmetic, cost-optimal thresholds, the macro→micro
   │                bridge, risk banding, attribution stability.
application/     Use cases. May use pandas; depends on domain protocols only.
   ▲                train / inference / scenario pipelines.
infrastructure/  Adapters. Implements the domain's ports.
   ▲                data, models, NLP, XAI, API, config, logging.
presentation/    Streamlit dashboard. Consumes the API.
```

**Why the domain is pure.** A model-risk reviewer has to be able to read how a
PD is floored, how a threshold is chosen and how a GDP shock becomes default
risk — without following a pandas call chain. `scripts/check_domain_purity.py`
runs in pre-commit and in the test suite; a stray `import numpy` in `domain/`
fails the build.

`domain/entities.py` holds `FEATURE_CONTRACT`, the single declaration of the
modelling dataset. The pandera schemas, the Pydantic API models and the
dashboard slider ranges are all **compiled from it**, so they cannot drift apart.

---

## What is actually implemented

### Leakage control

`TimeSeriesSplit` is deliberately not offered. With a twelve-month outcome
horizon on a quarterly panel, an observation either side of a fold boundary
contributes its label to both sides. Four leakage-aware splitters are provided
(`purged_kfold`, `blocked`, `expanding`, `sliding`) with configurable purge and
embargo windows.

The final test block is always the **latest** period, never a random sample.

### Calibration, honestly measured

A ranking model is not a PD model. Both Platt scaling and isotonic regression
are fitted and compared on expected calibration error, with the uncalibrated
model competing as a candidate — if neither helps, the original is kept.

The calibrator is fitted on a block the base model never saw, and scored on a
third, disjoint block. This matters: an earlier revision calibrated on the
evaluation rows and reported a validation PR-AUC of 0.91 against a test PR-AUC
of 0.30. Champion selection duly promoted the illusion. The split is now
`fit → calibration → evaluation → test`, and the calibrated model that gets
registered is the one whose metrics were reported.

### Cost-sensitive decisions

Missing a defaulter costs ten times a false alarm, so the 0.5 cut-off is wrong.
The threshold that minimises expected business cost is searched for explicitly
and travels with the model.

### Champion selection

Gates applied in policy order: PR-AUC (not ROC-AUC — at an 8% base rate ROC-AUC
flatters everything), then calibration error, then latency. Among survivors the
lowest business cost wins. If nothing clears every gate, the best PR-AUC is
returned **with the reasons attached** rather than silently.

### Drift, with a fail-safe

PSI and a two-sample KS test run before every batch. Policy is configurable:
`block` (default) refuses to score a severely drifted batch; `warn` scores and
attaches the severity. Both always alert.

Two guards prevent the alarm being trained away:

- Macro variables are excluded — they are constant within a scoring batch and
  variable across a pooled reference, so their PSI is enormous by construction.
- Batches below `min_batch_size` return no verdict, so a single-obligor request
  is never refused for "drift".

The reference is the **most recent validated block**, not a sample of all
history, so ordinary seasonal variation does not read as drift.

### Explainability

SHAP (TreeSHAP with a KernelSHAP fallback), LIME, permutation importance,
DiCE counterfactuals, attribution stability across seeds and across time, and
fairness by sector (demographic parity, equalised odds, per-group calibration).

Counterfactuals answer the question that matters operationally: *which ratios
must move, and by how little, to bring this PD to target* — with macro variables
excluded from the actionable set, because "raise GDP" is not recourse.

### Macro → micro bridge

Two stages, so model validation can challenge each separately:

1. **Real economy** — macro deltas move sector revenue via per-sector elasticities.
2. **Credit** — the revenue shock plus the financing channel moves default log-odds.

Under the shipped severely-adverse scenario this produces a 4.9× odds multiplier
for consumer discretionary against 1.7× for utilities. The arithmetic lives in
`domain/services.py` and is inspectable via `GET /scenarios`.

### Training–serving parity

Both pipelines route NLP feature construction through one shared enricher
(`infrastructure/nlp/enrichment.py`), backed by the feature store. This is not
incidental: an earlier revision enriched only in training, and the drift
detector correctly flagged the resulting skew.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Score one obligor or a batch, optionally with explanations |
| `POST /explain` | SHAP, LIME or counterfactual explanation |
| `POST /scenario` | Macro stress scenarios over a portfolio |
| `GET /scenarios` | Scenario catalogue and the sector transmission matrix |
| `GET /health` | Model readiness and drift configuration |
| `GET /metrics` | Prometheus exposition |

Errors are structured: `FinMLError` becomes a JSON body with `error_type`,
`message` and `context`, and the status code follows the category — 422 for a
domain violation, 503 for an infrastructure failure, 409 for a blocked drifted
batch.

```bash
curl -s localhost:8000/health
```

---

## Configuration

Everything is Hydra, validated at compose time against dataclasses in
`infrastructure/config/schemas.py`. A typo fails before a model is trained.

```bash
python -m application.train_pipeline model.tuning.n_trials=50 seed=7
```

```bash
python -m application.train_pipeline models@model=fast data.n_entities=300
```

The `configs/models/` group maps onto the `model` node via `models@model`, so
`models@model=fast` selects the CI profile.

---

## Testing and quality

| Command | What it checks |
|---|---|
| `make test` | 306 tests, 82% coverage (gate: 80%) |
| `make lint` | ruff — clean |
| `make typecheck` | mypy `strict` on 49 modules — clean |
| `make security` | bandit — zero findings |
| `make check` | all of the above |

Beyond ordinary unit tests, the suite asserts architectural and numerical
properties that are easy to break silently:

- the domain imports nothing outside the standard library;
- dependencies point inwards only;
- no `print` and no silent `except: pass` anywhere in `src/`;
- the focal-loss gradient **and Hessian** match finite differences;
- adapters satisfy `is_classifier()` — without it `CalibratedClassifierCV`
  reads the wrong probability column and inverts every calibrated PD;
- SHAP output is normalised across all four layouts vendors return;
- validation and test metrics stay consistent, which is what catches a
  reintroduced calibration leak.

---

## Notes on this build

Several defects surfaced during development that are worth recording, because
each would have been invisible in a demo:

- **sklearn mixin order.** `class X(BaseEstimator, ClassifierMixin)` makes
  `is_classifier()` return `False`; `CalibratedClassifierCV` then calibrates
  against the negative class. Every calibrated probability came out inverted
  (AUC 0.78 → 0.22) with nothing raised. The order must be
  `(ClassifierMixin, BaseEstimator)`.
- **LightGBM custom objectives.** `predict_proba` returns 1-D **raw logits**,
  not probabilities, while XGBoost returns proper `(n, 2)`. Unnormalised, the
  focal-loss path reports a "probability" of 1.58.
- **sklearn ≥ 1.9** removed `cv="prefit"`; calibration now needs
  `FrozenEstimator`. Both paths are supported.
- **MLflow 3** put the `file://` tracking backend into maintenance mode and it
  now raises on connect. The default is `sqlite:///mlflow.db`, which needs no
  server.
- **`StackingClassifier` cannot be used here** — it builds meta-features with
  `cross_val_predict`, which requires the folds to be a partition, and the
  purged splitters deliberately are not. The stack is assembled directly.

### Optional extras

The NLP stack (`torch`, `transformers`, `bertopic`, `spacy`) is optional:

```bash
make install-nlp
```

Without it, FinBERT falls back to a finance lexicon with negation handling, NER
to regex patterns, and BERTopic to seeded keyword matching. Every fallback is
flagged `is_fallback=True` on the feature-store record and in the logs — a
lexicon score is never presented as a transformer score.

### Real data sources

`data.source` selects `synthetic` (default), `fred`, `worldbank` or `edgar`.
The remote adapters retry with backoff and cache payloads. FRED and World Bank
supply macro history only and log that the obligor panel remains synthetic;
`edgar` raises rather than pretending, because turning raw XBRL into a
comparable ratio panel is a project in itself.

---

## Layout

```
configs/            Hydra config groups (data, features, models, nlp, drift,
                    scenario, xai, api, dashboard, tracking)
src/domain/         entities, value_objects, services, repositories, exceptions
src/application/    train / inference / scenario pipelines, ports
src/infrastructure/ data, models, nlp, xai, api, config, logging
src/presentation/   streamlit_app.py
tests/              unit, integration, data-contract
scripts/            check_domain_purity.py
```
