# LLM Wiki — Telco Churn Challenge

Language for the Etisalat/IBM Telco churn challenge effort inside this repo: a churn classifier, a natural-language chat pipeline over it, and its deployment. One canonical term per concept; the canonical term is what appears in specs, code, and docs.

## Language

### Prediction domain

**Churn**:
A customer discontinuing their telecom service within the dataset's observation window. The thing we predict.
_Avoid_: Leaving, cancelling, defection

**Churn classification**:
The task framing: predict the binary Churn label (Yes/No) for a customer. The challenge overview's phrase "churn rates" is not the task.
_Avoid_: Churn-rate prediction, churn estimation

**Churn probability**:
The classifier's output for one customer: P(churn = Yes), a number between 0 and 1 produced by the model — never by the LLM.
_Avoid_: Score, risk (unqualified)

**Churner**:
A customer whose Churn label is Yes. The minority class (≈26.5%).
_Avoid_: Positive (unqualified), leaver

**Operating point**:
The probability threshold at which churn probability becomes a Yes/No decision, chosen by F1-max subject to precision ≥ 0.55.
_Avoid_: Cutoff, decision boundary

### Chat pipeline

**Extract-only LLM**:
The open-source language model whose only job is converting a user's question into a structured feature request; it never generates numeric values or predictions itself.
_Avoid_: Chatbot LLM, assistant model

**Feature request**:
The structured, schema-validated JSON object the extract-only LLM emits: which features the question touches and what kind of answer is wanted.
_Avoid_: Tool-call payload, intent, query plan

**Prediction response**:
The final chat answer: the tool-computed probability (or explanation) string-interpolated into natural language. Numbers in text come only from tool output.
_Avoid_: Generated answer, LLM reply

**Slot accuracy**:
The share of curated chat-test queries whose extracted feature request matches the expected selection. Target ≥ 95%.
_Avoid_: Extraction accuracy, intent accuracy

**Schema validity**:
Whether a feature request parses against its schema. Grammar-constrained decoding guarantees 100%; a violation is a system bug, not a chat failure.

### Evaluation

**Leakage guardrail**:
A structural rule that keeps test-set information out of training: resampling fitted on train folds only, Total_Charges coerced to numeric before splitting, customer identifiers dropped at load.

**External validation**:
Scoring the frozen trained model on real datasets from other carriers/countries (UCI Iranian Churn, Orange Telecom) and reporting the AUC drop honestly.
_Avoid_: Transfer testing, cross-dataset eval

**Synthetic sensitivity**:
CTGAN-generated perturbations of the training distribution, used only for robustness ablation and demo filler — never as evidence of generalization.
_Avoid_: Synthetic validation, synthetic test set

**Published band**:
The metric range achieved by rigorous published studies on this exact dataset (e.g., AUC 0.84–0.88). The honest target zone; scores materially above it trigger a leakage audit.
_Avoid_: SOTA, best score

### Deployment

**Demo window**:
An interval in which the serverless GPU is intentionally kept warm for a live demonstration; outside it, the GPU scales to zero.
_Avoid_: Always-on, warm pool

**Serving runtime**:
The inference server hosting the extract-only LLM (vLLM in the cloud, Ollama in local development); the application layer is runtime-agnostic.
_Avoid_: Model server, backend (unqualified)

**Spec/data mismatch**:
A documented divergence between the challenge PDF and the attached CSV (e.g., PDF says "Postal check", data says "Mailed check"; a header column name carries a trailing space). Documentation states the PDF's claim and defers to the data.
_Avoid_: Bug (unqualified), discrepancy (unqualified)
