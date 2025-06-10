### Datasets preparation

This project involves the following datasets:

- [MMLU](https://huggingface.co/datasets/cais/mmlu) - multiple-choice QA split into many topics.
- [GSM8k](https://huggingface.co/datasets/openai/gsm8k) - arithmetical problems
- [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) - medical multiple-choice QA

`LM-Polygraph` assumes that the dataset that you use for evaluation is uploaded to Huggingface Hub formatted with the correct prompt. This means that the dataset that you pass to `LM-Polygraph` should already contain instructions to the model. Compare, for example, original MMLU dataset linked above and [this one](https://huggingface.co/datasets/LM-Polygraph/mmlu/viewer/empirical_baselines). Note that it already contains few-shot examples and instruction for the model about the task.

We need to do the same with the three datasets that we want to evaluate on. We will do it in a zero-shot setting, so no need to include few-shot examples into the prompt. We need two versions of each dataset uploaded to HF Hub: one prompted for immediate answer ("Output only the answer itself, nothing else...") and one prompted for step-by-step reasoning.

You can create these versions as subsets or separate datasets, it doesn't matter. Each dataset needs to contain only two features - Input (fully prepared LLM prompt with a question and instruction) and Target (the correct answer to compare model output with).

For GSM8k, note that we only need the answer itself for the target. We can discard the reasoning in the answer column. 
