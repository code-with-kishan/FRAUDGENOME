# FRAUDGENOME — Success Criteria, Metrics, and 90s Demo Script

This document records objective acceptance criteria, key metrics, and the 90-second demo steps required to call the project a "10/10" implementation of FRAUDGENOME.

## High-level success criteria
- Detect pre-fraud mule recruitment (early-warning) and map fraud rings with lifecycle staging.
- Produce auditable, plain-English investigation briefs with evidence sources (SHAP + Fraud DNA + graph context).
- Operate continuously with drift detection and automated retraining (human-in-loop for promotion).

## Target metrics (validation / production)
- AUC-ROC (ensemble): >= 0.87 on hold-out set of labeled cases.
- Precision-Recall AUC (ensemble): >= 0.72 on hold-out set.
- Early-warning DTW match recall: detect >= 70% of confirmed mules in the pre-activation window (validation era), with maximum false-positive rate tuned by CTI.
- Operational F1 at operating point: >= 0.55 (use PR-driven threshold selection).
- Cohort-relative FP reduction: >= 30% reduction vs. global baseline (measured on validation cohorts).
- Ring detection precision (community→confirmed-mule linking): >= 0.65 on labeled ring samples.

## Latency & UX targets (demo / production)
- CTI computation per account (single request, in-memory models): < 200 ms average.
- DTW-based matching (top-k against library): < 500 ms for prototype (optimizations TBD for scale).
- GenAI brief generation (PDF draft): < 30 seconds.

## Data & instrumentation requirements
- Labeled confirmed-mule flag (anchor): `F3924` present and timestamped.
- Time-series feature availability for anchors: `F321`, `F3836`, `F2082` (history window N days).
- Transaction-level timestamps to build backward windows and synchronized transition graphs.
- Ground-truth outcomes and analyst feedback logging (for retraining and champion-challenger).

## Concrete acceptance checks (per step)
- Fraud DNA library: store per-mule prototypes (npy or parquet) and metadata (account_id, pattern_window_start/end, pattern_id). Unit test: given a confirmed mule, extracted pattern must contain the labeled activation timestamp and at least 1 prior-day window.
- DTW matching: API returns top-3 nearest Fraud DNA patterns with distances. Unit test: distances are finite and ordering is stable.
- Ensemble & SHAP: training pipeline saves models and SHAP explainer; example account returns top-5 feature contributions with plain-English mapping.
- Ring topology: script outputs communities with stage label in {Recruiting, Active, Dispersing, Dormant}. Unit test: stage assignment heuristic applied consistently.
- GenAI brief: generated PDF includes sources and numeric values (no hallucinated facts). Unit test: brief contains SHAP top-5 and DTW match IDs.

## 90-Second Demo Script (exact sequence)
1. Start dashboard; real-time tiles show CTI distribution (5s). Presenter: "CTI gives a single actionable score per account." 
2. Click a flagged account (3–5s): show account panel with CTI=91, DTW match (Pattern #3, distance 0.23), and SHAP waterfall (top-3 reasons). Narration: "This account matches Fraud DNA Pattern #3 and scored 91 due to dormancy reactivation and velocity spike." 
3. Trigger ring map (5s): Node cluster animates; highlight Ring #7 (11 nodes) and show lifecycle stage: Recruiting. Narration: "Ring #7 is in Recruiting—no fraudulent transfers observed yet." 
4. Click "Generate Brief" (20s): PDF pops up with structured investigation brief, top-5 SHAP inputs, DTW evidence, recommended action, and FMR draft. Narration: "Here is a fully-sourced brief ready for compliance review." 
5. Open the signature library (15s): show confirmed mule patterns, coverage, and active prototype windows. Narration: "These stored FraudDNA patterns become reusable memory for future investigations." 

Timing note: run the demo locally with canned data for reproducibility; all times measured on a modern developer laptop and may vary in production.

## Next validation gates
1. Reproduce target metrics on a held-out labeled dataset (PR-AUC, AUC-ROC). 
2. End-to-end pipeline test: data ingestion → Fraud DNA extraction → model training → API scoring → brief generation.
3. Compliance audit checklist: brief contains only verifiable sources, with analyst sign-off flow.
