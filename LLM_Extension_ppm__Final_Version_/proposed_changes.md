# Proposed Text Replacements for `sample-manuscript.tex`

Two categories of changes:
1. **"Hash" -> "Anonymize"**: Replace all forms of "hash/hashed/hashing" with "anonymize/anonymized/anonymization"
2. **"Trained" -> "Provided with"**: When referring to LLMs being "trained" on traces, replace with "provided with" (benchmark models that are genuinely trained are left unchanged)

---

## Category 1: Hash -> Anonymize

### Line 649
**Current:**
> Based on it, a deterministic hash function $Hash: \MH \to \Sigma^4$ maps each $s \in \MH$ to a unique 4-character identifier ...

**Suggested:**
> Based on it, a deterministic anonymization function $Anon: \MH \to \Sigma^4$ maps each $s \in \MH$ to a unique 4-character identifier ...

---

### Line 651
**Current:**
> ...every $s \in \MH$ is replaced by $Hash(s)$. Then, the prediction quality in terms of MAE or F1-Score is compared between hashed and non-hashed prompts to quantify the semantic dependency of the LLM. Significant performance degradation under hashed prompts indicates that the LLM is relying on its embodied knowledge.

**Suggested:**
> ...every $s \in \MH$ is replaced by $Anon(s)$. Then, the prediction quality in terms of MAE or F1-Score is compared between anonymized and non-anonymized prompts to quantify the semantic dependency of the LLM. Significant performance degradation under anonymized prompts indicates that the LLM is relying on its embodied knowledge.

---

### Line 655
**Current:**
> `\includegraphics[width=.99\linewidth]{Figures/hashed_trace.pdf}`

**Suggested:**
> `\includegraphics[width=.99\linewidth]{Figures/anonymized_trace.pdf}`

*(Note: the PDF file itself would also need to be renamed)*

---

### Line 656
**Current:**
> \caption{Example of the Hashing procedure applied to the trace in the Figure~\ref{fig:seqenc}.}

**Suggested:**
> \caption{Example of the Anonymization procedure applied to the trace in the Figure~\ref{fig:seqenc}.}

---

### Line 657
**Current:**
> `\label{fig:enchashed}`

**Suggested:**
> `\label{fig:encanonymized}`

*(Note: all `\ref{fig:enchashed}` references in the document must also be updated)*

---

### Line 660
**Current:**
> ...the context-sensitive set $\MH$ is $\MH = \{Dept,\ D1, A1, A4\}$ since the variable name “Dept” can be associated to some organization-related context, as do its value and the activity names, while the activity durations $E1-S1$ are only numbers that do not need to be hashed. This procedure also removes the context-related information that has been added to the prompt (cf.\ bold sentences in Listing~\ref{lst:train}). We remind that this is only the hashing procedure for a given trace...

**Suggested:**
> ...the context-sensitive set $\MH$ is $\MH = \{Dept,\ D1, A1, A4\}$ since the variable name “Dept” can be associated to some organization-related context, as do its value and the activity names, while the activity durations $E1-S1$ are only numbers that do not need to be anonymized. This procedure also removes the context-related information that has been added to the prompt (cf.\ bold sentences in Listing~\ref{lst:train}). We remind that this is only the anonymization procedure for a given trace...

---

### Line 668
**Current:**
> Null hypothesis ($H_0$): No significant difference exists between prediction performance (MAE/F1-Score) of hashed and non-hashed prompts across use cases.

**Suggested:**
> Null hypothesis ($H_0$): No significant difference exists between prediction performance (MAE/F1-Score) of anonymized and non-anonymized prompts across use cases.

---

### Line 670
**Current:**
> Alternative hypothesis ($H_1$): Significant performance difference exists, with non-hashed prompts expected to outperform due to semantic exploitation.

**Suggested:**
> Alternative hypothesis ($H_1$): Significant performance difference exists, with non-anonymized prompts expected to outperform due to semantic exploitation.

---

### Line 690
**Current:**
> \caption{Comparison between LLM and LLM hashed prompts with Nemenyi Post-Hoc Test Results}

**Suggested:**
> \caption{Comparison between LLM and LLM anonymized prompts with Nemenyi Post-Hoc Test Results}

---

### Line 693
**Current:**
> \textbf{KPI} & \textbf{Use Case} & \textbf{LLM} & \textbf{LLM hashed} & \textbf{Difference} & \textbf{P-value} & \textbf{Significance} \\

**Suggested:**
> \textbf{KPI} & \textbf{Use Case} & \textbf{LLM} & \textbf{LLM anonymized} & \textbf{Difference} & \textbf{P-value} & \textbf{Significance} \\

---

### Line 710
**Current:**
> ...the difference between the hashed and non-hashed treatments. ...the difference is computed as hashed $-$ non-hashed, whereas for the F1-Score, it is computed as non-hashed $-$ hashed.

**Suggested:**
> ...the difference between the anonymized and non-anonymized treatments. ...the difference is computed as anonymized $-$ non-anonymized, whereas for the F1-Score, it is computed as non-anonymized $-$ anonymized.

---

### Line 712
**Current:**
> All comparisons reject the null hypothesis of no difference between hashed and non-hashed prompts, confirming the LLM's reliance on semantic embodied knowledge.

**Suggested:**
> All comparisons reject the null hypothesis of no difference between anonymized and non-anonymized prompts, confirming the LLM's reliance on semantic embodied knowledge.

---

### Line 938
**Current:**
> ...the context in the prompts was hashed and the experiments repeated...

**Suggested:**
> ...the context in the prompts was anonymized and the experiments repeated...

---

### Line 970
**Current:**
> In this Section is reported an example of an hashed input prompt...

**Suggested:**
> In this Section is reported an example of an anonymized input prompt...

---

### Line 972
**Current:**
> `%\label{lst:hashed_ao}`

**Suggested:**
> `%\label{lst:anonymized_ao}`

*(Note: all references to `lst:hashed_ao` must also be updated)*

---

## Category 2: "Trained" -> "Provided with" (LLM context only)

> **Rule:** Only change "trained" when the subject is the LLM. When benchmarks (CatBoost, PGTNet, etc.) are described as "trained", leave them unchanged since they are genuinely trained.

### Line 207 (Abstract)
**Current:**
> ...the LLM surpasses the benchmark methods, also reaching the same accuracy when benchmarks have been trained on the whole event log.

**No change needed** -- here "trained" refers to benchmarks, not the LLM.

---

### Line 270 (RQ1)
**Current:**
> When trained on event logs with a limited number of traces, do LLMs achieve superior prediction quality...

**Suggested:**
> When provided with event logs containing a limited number of traces, do LLMs achieve superior prediction quality...

---

### Line 537 (repeats RQ1)
**Current:**
> \textit{When trained on event logs with a limited number of traces, do LLMs achieve superior prediction quality...}

**Suggested:**
> \textit{When provided with event logs containing a limited number of traces, do LLMs achieve superior prediction quality...}

---

### Line 545
**Current:**
> We measured the accuracy of the LLM when trained on 100 traces sampled from $\ML^{comp}$...

**Suggested:**
> We measured the accuracy of the LLM when provided with 100 traces sampled from $\ML^{comp}$...

---

### Line 569
**Current:**
> ...we examined model performance when trained on 2 and 10 traces.

**Suggested:**
> ...we examined model performance when provided with 2 and 10 traces.

*(This sentence refers to the LLM from prior work)*

---

### Line 620
**Current:**
> ...the MAE when the models are trained on the full event log (\texttt{all\_df}) or on only 100 traces (\texttt{100 Traces}), respectively.

**Suggested (partial):**
This line refers to "models" generically (both LLM and benchmarks in the same table). Consider rephrasing:
> ...the MAE when the models use the full event log (\texttt{all\_df}) or only 100 traces (\texttt{100 Traces}), respectively.

---

### Line 630
**Current:**
> When restricted to 100 training traces, the LLM outperforms CatBoost...

**Suggested:**
> When restricted to 100 traces, the LLM outperforms CatBoost...

*(Removing "training" avoids implying the LLM is trained on them)*

---

### Line 632
**Current:**
> ...the LLM operates without any task-specific training, relying solely on the 100 example traces provided within the prompt.

**No change needed** -- this line already correctly distinguishes LLM from trained models.

---

### Line 860
**Current:**
> ...the LLM outperforms not only the derived $\beta$-learner in the scenario in which they are trained on 100 samples...

**Suggested:**
> ...the LLM outperforms not only the derived $\beta$-learner in the scenario in which they are provided with 100 samples...

*(Here "they" could refer to both the LLM and the beta-learners; the beta-learners are genuinely trained, but the LLM is not. Consider splitting the sentence for clarity.)*

---

### Line 862
**Current:**
> ...when the learners have been trained on the whole event log.

**No change needed** -- refers to beta-learners, not the LLM.

---

### Line 865
**Current:**
> ...the $\beta$-learner in the scenario in which they have been trained on 100 samples, but also when the learners have been trained on the whole event log...

**Suggested (first part only, about the LLM):**
> ...the $\beta$-learner in the scenario in which they are provided with 100 samples, but also when the learners have been trained on the whole event log...

---

### Line 866
**Current:**
> ...the LLM outperforms not only the derived $\beta$-learner in the scenario in which they have been trained on 100 samples, but also when the learners have been trained on the whole event log...

**Suggested:**
> ...the LLM outperforms not only the derived $\beta$-learner in the scenario in which they are provided with 100 samples, but also when the learners have been trained on the whole event log...

---

### Line 936
**Current:**
> RQ1 confirms the LLM's superiority when the model can only leverage 100 traces as training, outperforming state-of-the-art benchmarks for both KPIs.

**Suggested:**
> RQ1 confirms the LLM's superiority when the model is only provided with 100 traces, outperforming state-of-the-art benchmarks for both KPIs.

---

## Additional notes

- **Table labels** referencing "hash" (e.g., `\ref{tab:mae_hash}` on line 712) should be updated to `\ref{tab:mae_anonymized}`, and the corresponding `\label` definitions adjusted accordingly.
- **Figure/Listing files** named with "hashed" (e.g., `hashed_trace.pdf`, `lst:hashed_ao`, `lst:hashed_ao_answ`) should be renamed and their references updated.
- **Line 662** already uses the word "anonymized" (`replaced with their anonymized representation`), which is consistent with the proposed changes.
- **Line 543** contains an important disclaimer about LLM "training" terminology. After applying the changes above, this disclaimer may no longer be needed, or could be simplified.
