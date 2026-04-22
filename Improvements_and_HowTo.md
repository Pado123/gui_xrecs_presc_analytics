# Improvements to Address Reviewer Concerns — Practical Guide

This document maps each major reviewer concern to a concrete, implementable improvement.

---

## I1. Test at Least One Additional LLM (addresses M1)

**Why it matters:** All results rely on Gemini 2.5 Flash Thinking. Generalizability claims require replication on at least one architecturally different model.

**What to do:**
- Select one open-source model (e.g., Llama 3.1 70B or Qwen2.5 72B via Ollama or HuggingFace) and one alternative proprietary API (e.g., GPT-4o-mini for cost reasons).
- Run the identical experimental pipeline (RQ1 only, all 6 use cases, 20 subsamples of 100 traces) on both additional LLMs.
- Report results in a new table alongside Gemini.
- If the pattern holds across models → generalizability is supported. If it doesn't → qualify claims accordingly.

**Practical effort:**
- The existing code (`gui_xrecs_presc_analytics` repo) already wraps the LLM call; swapping the model is a configuration change.
- Llama 3.1 70B can be run locally if a GPU is available, eliminating API cost concerns.
- If full replication is too costly, run at least the Bpi12 use case on GPT-4o-mini as a representative cross-model check and acknowledge the limitation for the others.

**Key code location:** The LLM call in the main pipeline; add a `model_name` parameter that routes to different API clients.

---

## I2. Fix the Benchmark Comparison Framing (addresses M2)

**Why it matters:** Claiming "LLM surpasses benchmarks" when PGTNet trained on full data beats the LLM on Bpi12 is scientifically inaccurate.

**What to do:**
- In the abstract and results sections, replace "surpasses" with "matches or outperforms benchmarks trained on 100 traces, and approaches the performance of models trained on the full dataset."
- Add a dedicated paragraph in Section 5.1.2 that explicitly discusses the cases where the full-data benchmark wins.
- Add a table column or footnote that flags statistical significance for RQ1 comparisons (see I7 below).
- In the conclusion, state the nuanced finding: "the LLM is competitive with, but does not universally replace, models trained on large logs."

**Practical effort:** Writing change only. No new experiments required.

---

## I3. Fix the RQ2 Anonymization Confound (addresses M3)

**Why it matters:** The current anonymization removes both semantic strings AND the background text instructions simultaneously. These are two different interventions, and the results cannot distinguish their effects.

**What to do:** Add a third experimental condition:

| Condition | Activity names hashed | Background text removed |
|---|---|---|
| (a) Full prompt | No | No |
| (b) Strings only anonymized | Yes | No |
| (c) Full anonymization (current) | Yes | Yes |

- Condition (b) is the missing control. Run it for all 6 use cases.
- Compare (a) vs (b) to isolate semantic exploitation from activity names.
- Compare (b) vs (c) to isolate the contribution of the background text.
- Apply the Nemenyi test to all three pairwise comparisons.

**Practical effort:**
- Condition (b) requires modifying the anonymization function to hash context-sensitive strings but preserve the background text block (lines 36-38 of Listing 1).
- In the existing code, this means applying `Anon(s)` to the `ActTimeSeq` and attribute values but keeping the bold-formatted background description unchanged.
- Run the same 20-subsample protocol already implemented for condition (c).

---

## I4. Add Inter-Rater Reliability to β-learner Coding (addresses M4)

**Why it matters:** The β-learner taxonomy is derived from subjective reading of 50 reasoning traces per KPI. Without reliability measures, it is not scientific.

**What to do:**
- Have a second rater (a co-author or a graduate student not involved in the initial coding) independently read a random subset of 30 reasoning traces per KPI (out of the 50 analyzed) and assign each to a β-learner category using a written codebook.
- Compute Cohen's κ between the two coders.
- If κ > 0.70 (substantial agreement), report it as evidence of reliability.
- If κ < 0.70, either refine the codebook and re-code, or merge ambiguous categories.
- Add the codebook as a supplementary appendix.

**Practical effort:**
- The 50 reasoning traces per KPI have already been read; they need to be saved as structured text.
- The codebook can be a 1-page document listing each β-learner with a definition and two representative example phrases.
- Cohen's κ computation takes <1 hour with `sklearn.metrics.cohen_kappa_score`.

---

## I5. Validate β-learner Re-implementations (addresses M5)

**Why it matters:** If the β-learner re-implementations are poorly tuned, showing the LLM outperforms them proves nothing.

**What to do:**
- For each β-learner, perform a basic hyperparameter search on the validation set (using the 10% validation split already defined in Section 5.1.1).
  - For KNN-based learners: tune `k` over {3, 5, 10, 15, 20}.
  - For tree-based learners: tune `max_depth` over {3, 5, 10} and `n_estimators` over {50, 100, 200}.
- Report the best-validated β-learner results alongside the current ones.
- Add a sentence: "β-learner hyperparameters were tuned on the validation set to ensure competitive baselines."

**Practical effort:**
- The `beta_learner_eval.py` file already exists at `llmp/analysis/beta_learner_eval.py`.
- Add a `GridSearchCV` or manual loop over the hyperparameter grid; the infrastructure is already in place.
- Expected additional compute: ~2-4 hours on a standard machine.

---

## I6. Add Cost, Latency, and Scalability Analysis (addresses M6)

**Why it matters:** TMIS readers need to know if this approach is deployable. An approach that costs $5 per prediction or takes 30 seconds per trace is not a management recommendation.

**What to do:**
- **Cost:** For each of the 20 experimental runs, log the token counts (prompt tokens + completion tokens) using the API's usage metadata. Compute average and total cost at current Gemini 2.5 Flash Thinking pricing. Report as "average cost per prediction."
- **Latency:** Log wall-clock time per LLM API call. Report median inference time.
- **Comparison:** Report CatBoost and PGTNet training time (one-time cost) and prediction latency for reference.
- **Scalability discussion:** Add a paragraph noting that context length constrains the approach to ~100 traces at current pricing, and that this constraint is architectural, not just a design choice.

**Practical effort:**
- Gemini API responses include `usage_metadata` with token counts; add a counter in the existing API wrapper.
- Latency logging: wrap each LLM call with `time.time()` before and after.
- No new experiments needed; this can be retroactively estimated from existing runs if the logs were saved, or by running a small timing experiment (e.g., 5 runs on Bpi12).

**Example table to add:**

| Model | Avg. Training Cost | Avg. Prediction Latency | Cost per Prediction |
|---|---|---|---|
| CatBoost (100 traces) | ~5s (CPU) | <1ms | ~$0.00 |
| PGTNet (100 traces) | ~10min (GPU) | ~5ms | ~$0.00 |
| LLM (Gemini 2.5 Flash) | N/A (pre-trained) | ~3-10s | ~$0.01-0.05 |

---

## I7. Add Statistical Tests for RQ1 (addresses M7)

**Why it matters:** The core research question (RQ1) has no formal significance test. Overlapping confidence intervals (e.g., Bpi12: LLM 6508 ± 235 vs. CatBoost full 6846) could easily be non-significant.

**What to do:**
- Apply the Wilcoxon signed-rank test (the same test used for RQ3) to compare LLM vs. each benchmark across the 20 experimental subsamples.
- Specifically test: (a) LLM vs. CatBoost-100traces, (b) LLM vs. PGTNet-100traces (for total time).
- Report p-values and significance markers (*, **, ***) in Tables 2 and 3.
- Also test: LLM-100traces vs. CatBoost-fulldata to explicitly characterize where the LLM falls short.

**Practical effort:**
- The 20-subsample results are already stored; run `scipy.stats.wilcoxon(llm_results, baseline_results)`.
- This takes <1 hour of scripting.

---

## I8. Expand Dataset Coverage (addresses M8)

**Why it matters:** Three datasets, one of which is confidential, limits reproducibility and scope.

**What to do (minimum viable):**
- Add at least one more public PPM benchmark: SEPSIS (hospital event log, widely used, public), BPIC 2017, or BPIC 2019 (purchase orders). These are available via the 4TU Research Data repository.
- If adding a full new dataset is too costly, at minimum, add results for SEPSIS as it is the most-used healthcare PPM benchmark and would validate the Hospital results on a public dataset.
- Acknowledge the Hospital dataset's confidentiality explicitly in a threat-to-validity section and note that SEPSIS serves as a partial public substitute.

**Practical effort:**
- SEPSIS is a 1000-trace log; downloading and preprocessing it using the existing pipeline is ~1 day of work.
- The preprocessing pipeline (`utils/`) appears general enough to handle it.

---

## I9. Add Non-Determinism Controls (addresses m5)

**Why it matters:** LLM outputs are stochastic. The 20-subsample protocol captures training data variability but not output variability for the same input.

**What to do:**
- Set temperature to 0 (greedy decoding) in all Gemini API calls to maximize reproducibility. Document this in the experimental setup.
- If temperature 0 is not feasible (some models don't support exact greedy), use a fixed seed if the API supports it, and document the setting used.
- Add a sentence in Section 5.1.1: "To control for LLM output variability, all API calls were made with temperature=0."
- Optionally, run a small experiment (5 traces × 5 calls at temperature=0 and temperature=1) to quantify within-run variability and show it is small relative to across-sample variability.

---

## I10. Deepen the Management/IS Discussion (addresses m6)

**Why it matters:** TMIS expects a clear articulation of implications for practitioners and IS theory.

**What to do:** Add a 1-page "Discussion: Implications for Practice and Research" section before the conclusion. Structure it as:

1. **For process mining practitioners:** LLM-based PPM enables deployment in organizations that cannot collect large event logs (e.g., rare processes, new process variants, regulated industries). No retraining is needed when the process changes. The cost-per-prediction model shifts from upfront training investment to per-inference API costs.

2. **For IS researchers:** The β-learner framework provides a novel methodology for explainable process predictions and offers a bridge between symbolic (rule-based) and neural approaches to PPM. The anonymization experiment methodology could generalize to testing semantic exploitation in other LLM-based IS applications.

3. **Limitations for managers:** Current context length limits practical scale to ~100 historical traces. API dependency introduces vendor lock-in and data privacy risks (process data sent to external APIs). These should be weighed against the zero-training-data advantage.

---

## I11. Ablate Prompt Components (addresses m4)

**Why it matters:** The 7-component prompt is presented without validation. It is unclear which components are necessary.

**What to do (optional but recommended):**
- Run a 2×2 ablation: with/without domain background text (lines 36-38) × with/without trace attribute description (lines 8-9 of Listing 1).
- Report results on Bpi12 only (as the representative case).
- This directly addresses the concern about optional components and strengthens the prompt design claims.

**Practical effort:**
- 4 conditions × 20 subsamples × 1 dataset = 80 additional API calls.
- Minor code change: add flags to the prompt construction function.

---

## Priority Order for Revision

| Priority | Item | Effort | Impact |
|---|---|---|---|
| Critical | I3 (fix RQ2 confound) | Medium | High — invalidates current RQ2 interpretation |
| Critical | I7 (stats for RQ1) | Low | High — basic methodological gap |
| Critical | I4 (β-learner reliability) | Low-Medium | High — invalidates RQ3 taxonomy |
| High | I1 (additional LLM) | High | High — generalizability |
| High | I6 (cost/latency) | Low | High — TMIS relevance |
| High | I2 (comparison framing) | Low | Medium — writing fix |
| Medium | I5 (β-learner tuning) | Medium | Medium — baseline quality |
| Medium | I8 (more datasets) | High | Medium — breadth |
| Medium | I10 (IS discussion) | Low | Medium — journal fit |
| Optional | I9 (temperature=0) | Low | Low — reproducibility |
| Optional | I11 (prompt ablation) | Medium | Low — supplementary evidence |
