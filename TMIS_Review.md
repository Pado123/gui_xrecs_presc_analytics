# Peer Review for ACM Transactions on Management Information Systems (TMIS)

**Manuscript:** "Exploring LLM Features in Predictive Process Monitoring for Small-Scale Event-Logs"  
**Authors:** Alessandro Padella, Massimiliano de Leoni, Marlon Dumas  
**Reviewer Decision:** Major Revision

---

## Summary

This paper extends prior work by the same group on LLM-based Predictive Process Monitoring (PPM), introducing three research questions around prediction quality in data-scarce settings, semantic exploitation through anonymization experiments, and the derivation of so-called β-learners — re-implementable reasoning patterns extracted from LLM outputs. Experiments are conducted on three event logs (Bpi12, Bac, Hospital) across two KPIs (total time, activity occurrence) using Gemini 2.5 Flash Thinking, benchmarked against CatBoost and PGTNet.

The topic is timely and the research questions are relevant. However, the paper suffers from several methodological, validity, and presentation issues that, in their current form, prevent acceptance.

---

## Assessment of Contribution

The main contributions claimed are: (i) demonstration that an LLM with 100 in-context traces matches or outperforms state-of-the-art baselines trained on the full dataset; (ii) empirical evidence that LLMs leverage semantic domain knowledge; (iii) identification of β-learner reasoning patterns and their re-implementation.

Contribution (i) extends [29] from regression to both regression and classification — a non-trivial extension. Contributions (ii) and (iii) are novel. However, contribution (i) is overstated in several places, and the scientific rigor underpinning contributions (ii) and (iii) requires strengthening before they can be accepted as solid findings.

---

## Major Concerns

### M1. Single LLM Tested — Severe Generalizability Issue

The entire empirical evaluation rests on one proprietary model, Gemini 2.5 Flash Thinking. No experiments are reported with any other LLM (GPT-4o, Claude 3.5, Llama 3, Mistral, or even older Gemini versions). This is a fundamental threat to external validity. The performance advantages observed — particularly the semantic leverage effect in RQ2 — may be specific to Gemini's training corpus or its particular chain-of-thought architecture. The paper acknowledges in the future work section that "performance of upcoming models may see fast improvements," but this implicitly admits the results are model-specific. Without at least one replication on a second LLM of a different architecture (open-source vs. proprietary, different vendor), the main findings cannot be generalized. TMIS readership expects findings to be representative of the technology class, not of a single product.

### M2. Unfair Framing of Benchmark Comparisons

The framing in the abstract and results sections (e.g., "the LLM surpasses the benchmark methods") is misleading. Specifically:

- For Bpi12 total time, PGTNet trained on the full dataset (3888 MAE) substantially and significantly outperforms the LLM (6508 MAE). This is hidden behind the aggregate narrative.
- For Hospital total time, the LLM does beat both benchmarks even at full training — but this is a special case arguably attributable to the richer trace attributes in that dataset that CatBoost and PGTNet cannot semantically interpret. It should be presented as such.
- The comparison between an LLM using 100 in-context traces (not trained) and a supervised model trained on 100 traces is methodologically questionable. The LLM has been pre-trained on vast corpora that may include process-related data. The paper acknowledges this in bold on page 11 but then proceeds to treat the two systems as if they were under equivalent constraints. This acknowledgment is insufficient — the pre-training data overlap should be discussed as a confound.

### M3. RQ2 Anonymization Experiment Has an Uncontrolled Confound

The anonymization procedure replaces all context-sensitive strings with 4-character random identifiers AND simultaneously removes the domain-specific background text (the bold sentences in Listing 1). These are two distinct interventions. The observed performance degradation (e.g., 1702% MAE increase in Hospital) could be caused by the removal of the background text instructions alone, the replacement of activity names, or a combination. No controlled condition isolates these effects. A proper design would require at least three conditions: (a) full prompt, (b) anonymized strings but background text preserved, (c) both anonymized and background removed. As currently designed, RQ2 does not isolate semantic exploitation from the effect of removing instructional context.

### M4. β-learner Derivation is Subjectively Coded Without Reliability Measures

The β-learners are derived by manually reading 50 LLM output reasoning traces per KPI, totaling 150 traces across all configurations. The paper provides no inter-rater reliability measure (e.g., Cohen's κ), no description of whether multiple annotators independently coded the traces, and no formal codebook for the classification. The categorization of reasoning patterns into families (knn act, knn att, time seq, path pred for regression; Activity-Based, State-Based, Att-Based, Positive Evidence for classification) is presented as objective fact, but it is the product of a subjective interpretive exercise. Without reliability validation, it is unclear whether another researcher would arrive at the same taxonomy. This undermines the scientific status of contribution (iii).

### M5. Re-implementation Quality of β-learners is Not Independently Validated

The β-learners are re-implemented as KNN or tree-based algorithms. However, the paper provides no evidence that these re-implementations are faithful or well-tuned. The comparison between the LLM and each β-learner (Tables 5 and 6) is used to argue that "the LLM performs more complex analysis than each individual β-learner." But if the β-learner implementations are suboptimal (e.g., poor hyperparameter choices, arbitrary feature engineering), the comparison is meaningless — the LLM would trivially outperform a poorly tuned baseline. The paper should show that each β-learner is a competitive implementation of its corresponding strategy, e.g., by tuning it to match or approach the LLM's performance when the LLM is using that strategy, or at minimum by validating that each β-learner's full-data performance is competitive with CatBoost.

### M6. No Cost, Latency, or Scalability Analysis

For a paper targeting TMIS — a journal focused on management information systems — the practical deployment dimensions are entirely absent. No API cost analysis is provided. No inference latency comparison. No discussion of what happens as the context window fills (the paper notes context length is a bottleneck, yet the implications for scalability are not addressed). No analysis of what fraction of predictions require retrying due to malformed outputs. In a business process management context, a technique that takes 10× longer or costs 100× more than a CatBoost model is not a viable recommendation regardless of marginal accuracy gains. This is a critical omission for TMIS.

### M7. No Statistical Testing for RQ1

For RQ2, the Nemenyi post-hoc test is employed. For RQ3, the Wilcoxon signed-rank test is used. Yet for RQ1 — arguably the central research question — only mean ± standard deviation is reported, with no formal statistical comparison between the LLM and benchmarks. Given that confidence intervals frequently overlap (e.g., Bpi12 total time: LLM 6508 ± 235 vs. CatBoost 9394 ± 1275), the absence of formal tests makes it impossible to assess whether the LLM's advantage is statistically significant or within noise. This is a basic methodological gap.

### M8. Dataset Limitations and Reproducibility

Only three datasets are used, one of which (Hospital) cannot be published due to confidentiality constraints. This means a key dataset used to support the strongest claims (Hospital is the only case where the LLM beats all benchmarks even on full training data) cannot be independently replicated. Furthermore, only one dataset (Bpi12) is a standard public benchmark widely used in the PPM community. Bac is semi-public. The limited dataset diversity, particularly the absence of any dataset with highly variable trace lengths, complex event attributes, or non-Western-language activity names, leaves the boundary conditions of the approach unknown.

---

## Minor Concerns

### m1. Grammatical and Stylistic Issues

The paper contains several grammatical errors and awkward constructions that a careful proofread would fix. Examples: "implying implying" (page 10, line 497); "we examined model performance when provided with 2 and 10 traces.." (double period, page 11); "This procedure tries to mimic the reality at time $t_{split}$" reads awkwardly; "Since this paper targets data-scarce environments, we simulate a scenario" — the paper targets but the authors simulate is a conflation. More careful editing is required before publication.

### m2. Definition of "Small-Scale" is Informal

The paper uses "small-scale" in the title and throughout without formally defining what constitutes a small-scale log. The threshold of 100 traces appears pragmatically chosen due to context length limitations rather than from a theoretically motivated boundary. This should be acknowledged and, if possible, the sensitivity to this threshold explored (e.g., 50, 100, 200 traces).

### m3. Figure 5 is Difficult to Interpret

The convergence plots in Figure 5 contain many overlapping curves with a legend that is difficult to read. The captions do not sufficiently explain what "convergence" means in this context, nor what the dashed reference lines represent at a glance. A cleaner presentation, possibly separating the total time and activity occurrence subplots into separate figures, would improve readability.

### m4. Prompt Engineering Choices Not Ablated

The 7-component prompt structure (Section 4.2) is presented as a given. No ablation study tests whether removing or modifying individual components changes performance. In particular, the "domain-specific background information" component (Listing 1, lines 36-38) is described as optional, but its contribution is never measured in isolation. This is related to concern M3 above but applies more broadly to prompt design.

### m5. LLM Non-Determinism Not Fully Addressed

The paper acknowledges on page 9 that "reasoning can differ remarkably from one execution to another." Averaging over 20 random subsamples of the 100-trace log partially captures variability in the training data, but it does not control for variability in the LLM's stochastic outputs given the same input. The temperature or sampling parameters used are never reported. For reproducibility, these must be documented. If temperature was set to 0 (greedy), this should be stated; if not, then within-run variability has not been accounted for.

### m6. Management and IS Contribution Underdeveloped

For TMIS specifically, the paper should articulate the implications for business process management practitioners, IT managers, and IS researchers. The potential for non-technical users to apply LLM-based PPM without model training is a legitimate management-relevant contribution, but this is mentioned only in passing in the introduction and conclusion. A dedicated discussion section on the managerial, organizational, and IS-theoretical implications would better align the paper with TMIS scope and readership.

### m7. Related Work Coverage

The related work section is thin on recent relevant work. In particular, it lacks discussion of: (a) papers comparing LLMs with traditional ML on structured tabular data more broadly (e.g., TabLLM, LIFT), which provide important context for understanding when LLMs outperform classical methods on structured inputs; (b) uncertainty quantification in LLM predictions, which is directly relevant to deploying these systems in process monitoring; (c) retrieval-augmented generation approaches to process monitoring, beyond the brief mention of [6].

---

## Recommendation

**Major Revision.** The paper addresses a relevant problem and presents genuinely novel experiments. However, concerns M1 (single LLM), M2 (comparison framing), M3 (RQ2 confound), M4–M5 (β-learner methodology), M6 (no cost/latency), and M7 (no statistical test for RQ1) collectively prevent acceptance in the current form. A revised version that addresses these concerns — particularly by testing at least one additional LLM, correcting the experimental design of RQ2, and providing statistical tests for RQ1 — would be a substantially stronger contribution.

---

*Reviewed for ACM Transactions on Management Information Systems (TMIS)*
