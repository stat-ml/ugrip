# Internship Task Description: The Effect of Reasoning on Uncertainty Estimation for LLMs

## 🔍 Project Goal

This internship project investigates how the presence of **reasoning** in LLM outputs affects the performance of different **Uncertainty Estimation (UE)** methods. You will use the **LM-Polygraph** framework to benchmark multiple UE techniques on three popular QA datasets. Your focus will be on understanding the interplay between prompt structure, reasoning content, and the behavior of uncertainty metrics.

## 📚 Datasets

You will work with the following QA datasets:
- **GSM8K**: Arithmetic and reasoning-based grade-school questions.
- **MMLU**: Broad knowledge multiple-choice questions.
- **MedMCQA**: Medical domain multiple-choice questions.

For each dataset, prepare **two prompt versions**:

### 1. Direct Answer Version
- Prompt should **explicitly instruct the model to output only the final answer** with no explanation or reasoning.
- Ensure that **the model complies with this instruction**. If not, iterate on the prompt until it reliably outputs *just the answer string* (e.g., `42`, `C`, etc.).

#### Example (GSM8K):
> **Prompt:** What is 15 multiplied by 12? Please output only the final answer and nothing else.  

### 2. Reasoning-Inducing Version
- Prompt should **instruct the model to reason step by step**, but keep the reasoning to a **single paragraph**.
- Instruct the model to **output the answer at the end**, using a delimiter such as `### Answer:` to clearly separate reasoning from answer.
- Do **not** include the final answer in the prompt — only the format specification.

#### Example (GSM8K):
> **Prompt:** Solve the following math problem step by step. Keep your reasoning to one paragraph. At the end, clearly state the final answer using the format: `### Answer:`  
> What is 15 multiplied by 12?  

Once both versions are created:
- Format as two-column datasets:
  - Column 1: Full model input (prompt), without chat template.
  - Column 2: Target output (correct answer string).
- Upload each dataset version to HuggingFace Hub under a shared organization or user namespace.

## 🧪 Benchmarking with LM-Polygraph

Use the LM-Polygraph framework to benchmark each dataset version using three UE method families:

### A. Information-Theoretic Methods
- MSP, PPL, MTE, TokenSAR
- Metrics are aggregated over the full generated output.

### B. Consistency-Based Methods
- NumSemSets, DegMat, EigValLaplacian...

### C. Hybrid Methods
- SentenceSAR, SAR, SemanticEntropy

## 🧩 Deeper Analysis

Once initial benchmarking is complete, perform fine-grained experiments:

### 1. Information-Theoretic: Segment-Level Aggregation

For outputs with reasoning, decompose the model completion into:
- **Reasoning segment**: everything before the delimiter `### Answer:`
- **Answer segment**: everything after the delimiter

Perform uncertainty aggregation:
- Over **entire output** (baseline)
- Over **reasoning only**
- Over **answer only**

#### Example (MMLU):
> **Model output:**  
> The question is about basic electricity. Ohm’s law relates voltage, current, and resistance. Given that voltage = 10V and resistance = 2Ω, we compute current as I = V / R = 10 / 2 = 5A.  
> ### Answer: C  
>
> **Reasoning segment:** all before `### Answer: C`  
> **Answer segment:** `C`  
> Measure token-level entropy and logprobs over each part separately.

### 2. Consistency/Hybrid: Answer-Only Similarity

In reasoning-inducing prompts:
- Sample multiple model outputs.
- For each pair, compute similarity:
  - **Full-output similarity** (default)
  - **Answer-only similarity**, extracted via delimiter

Compare how these different similarity scopes affect:
- Uncertainty estimates
- Correlation with correctness
- Filtering performance

#### Example (MedMCQA):
> **Sample 1:**  
> Let's break down the question. The symptom profile matches a classic case of hypothyroidism...  
> ### Answer: B  
>
> **Sample 2:**  
> Based on the clinical signs and the standard diagnostic pathway...  
> ### Answer: B  
>
> Compare:
> - Full output similarity (includes varied reasoning)
> - Answer-only similarity (`B` vs `B`) — identical in this case

## ✅ Deliverables

1. Six datasets (three tasks × two prompt types), uploaded to HuggingFace Hub
2. LM-Polygraph benchmark results for each dataset version and UE method
3. Code and config files for running LM-Polygraph on all variants
4. Segment-aware UE analysis scripts
5. A concise report or presentation with the following:
   - Comparative results: PRR scores of UE methods under reasoning prompt vs direct answer prompt
   - Comparative results: PRR scores of UE methods under full-output vs segment-specific aggregation
   - Key findings about the effect of reasoning on uncertainty behavior
