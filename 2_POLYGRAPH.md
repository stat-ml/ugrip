<!--
This document is a Markdown guide for developers who want to extend
the LM-Polygraph benchmarking framework with custom methods.
-->
# Extending Benchmarking in LM-Polygraph

This guide provides an overview of LM-Polygraph's inner structure
and explains how to plug in custom benchmarking components.

## 1. Entry Point: `polygraph_eval`

All benchmarks start from the `polygraph_eval` script:

```bash
#!/usr/bin/env python3
...
@hydra.main(...)
def main(args):
    ...

if __name__ == "__main__":
    main()
```
【F:scripts/polygraph_eval†L1-L12】【F:scripts/polygraph_eval†L230-L233】

Runtime steps in `main`:
1. Load Hydra YAML config.
2. Setup logging and optional W&B.
3. Loop over random seeds.
4. Load model and dataset.
5. Build **stat calculators** _before_ UE estimators.
6. Initialize `UEManager` to run generation, compute stats, and calculate metrics.
7. Save results per seed.

> **Note:** Stat calculators run prior to estimators, as many estimators
> depend on statistics they produce.

## 2. Configuration via Hydra YAML files

Configurations reside under `examples/configs/` (standard) and
`examples/configs/instruct/` (instruction-finetuned model benchmarking). They compose on base
presets using Hydra's `defaults` mechanism.

### 2.1 Root presets (examples/configs)

The core YAML (e.g. `polygraph_eval_coqa.yaml`) declares key defaults:
- Default model, estimators, and stat_calculators presets
- Task settings (dataset, splits, batch_size, seeds, etc.)
- Base processing hooks for output normalization
【F:examples/configs/polygraph_eval_coqa.yaml†L9-L20】

Base processing injects normalization via `process_output_fn`
and `process_target_fn`:
【F:examples/configs/base_processing_coqa.yaml†L1-L6】

### 2.2 Instruction-finetuned model presets (examples/configs/instruct)

Configs for benchmarking instruction-finetuned models reuse root defaults and set `instruct: true`.
Chain-of-thought and blackbox estimator presets live here:
- `polygraph_eval_coqa_default_instruct.yaml`
- `cot_processing_coqa.yaml`
- `default_blackbox_estimators.yaml`
【F:examples/configs/instruct/polygraph_eval_coqa_default_instruct.yaml†L1-L15】【F:examples/configs/instruct/cot_processing_coqa.yaml†L1-L7】【F:examples/configs/instruct/default_blackbox_estimators.yaml†L1-L15】

## 3. Core Component Types

This section outlines the three core types in LM-Polygraph:

### StatCalculator

`StatCalculator` classes produce intermediate statistics required by uncertainty
estimators and generation metrics. Each calculator encapsulates a modular piece of
logic that transforms existing stats into new ones, possibly invoking the model.

```python
class StatCalculator(ABC):
    @abstractmethod
    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: Model,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """Calculate and return new statistics based on dependencies and model."""
```
【F:src/lm_polygraph/stat_calculators/stat_calculator.py†L9-L53】

**Key points:**
- `meta_info()` declares `stats` produced and `dependencies` required.
- `_register` in `register_default_stat_calculators` wraps each class into a
  `StatCalculatorContainer` with builder and default config.
- Called in topologically sorted order by `UEManager.calculate`.

### Estimator

`Estimator` classes consume completed statistics to output uncertainty scores
(`float` per sequence or token). They form the core uncertainty estimation methods.

```python
class Estimator(ABC):
    @abstractmethod
    def __call__(
        self,
        stats: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute uncertainty scores based on precomputed stats."""
```
【F:src/lm_polygraph/estimators/estimator.py†L8-L50】

**Key points:**
- Initialized with `stats_dependencies` (which stats to read) and `level` ('sequence', 'token', or 'claim').
- Invoked after all stat calculators have populated `batch_stats` in `UEManager.estimate`.

### GenerationMetric

`GenerationMetric` classes compute quality of model outputs by comparing
them to reference texts.

```python
class GenerationMetric(ABC):
    @abstractmethod
    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str]
    ) -> np.ndarray:
        """Compute reference uncertainty metric from stats and targets."""
```
【F:src/lm_polygraph/generation_metrics/generation_metric.py†L8-L60】

**Key points:**
- Initialized with `stats_dependencies` and `level` to match estimator outputs.
- Run in `UEManager.__call__` immediately after `Estimator` runs,
  feeding both `batch_stats` and `target_texts`.

## 4. Runtime Pipeline in `polygraph_eval`

Pseudocode:
```text
for seed in args.seed:
    model = get_model(args)
    dataset = Dataset.load(...)
    stat_cals = get_stat_calculator_names(args)
    estimators = get_ue_methods(args, model)
    gen_metrics = get_generation_metrics(args)
    ue_metrics = get_ue_metrics(args)

    manager = UEManager(
        data=dataset,
        model=model,
        estimators=estimators,
        builder_env_stat_calc=BuilderEnvironmentStatCalculator(model),
        available_stat_calculators=stat_cals,
        generation_metrics=gen_metrics,
        ue_metrics=ue_metrics,
        processors=[Logger()],
        ...
    )
    manager()
    manager.save(...)
```
【F:scripts/polygraph_eval†L131-L169】

### 4.1 Deep Dive: Model & Dataset Loading

- **Model loading**: `get_model(args)` dispatches to `get_whitebox_model`,
  `get_vllm_model`, or `get_blackbox_model` based on `args.model.type`.
  Whitebox paths invoke an external load script or the Transformers API to
  build `base_model` and `tokenizer`, then wrap them in a `WhiteboxModel`.
  【F:scripts/polygraph_eval†L291-L336】
- **Dataset loading**: `Dataset.load(...)` selects between CSV or HF datasets
  via `Dataset.from_csv`/`Dataset.from_datasets`, returning a batched
  `Dataset` that yields `(input_texts, target_texts)` tuples.
  【F:src/lm_polygraph/utils/dataset.py†L218-L276】【F:src/lm_polygraph/utils/dataset.py†L22-L52】

#### 4.1.1 WhiteboxModel: tokenization & generation

- **Construction**: `WhiteboxModel.from_pretrained(...)` loads a HuggingFace
  model and tokenizer (AutoModelForCausalLM/Seq2SeqLM, AutoTokenizer), sets
  `pad_token` if missing, and wraps them plus a `GenerationParameters` object
  into a `WhiteboxModel`.
  【F:src/lm_polygraph/utils/model.py†L630-L703】【F:src/lm_polygraph/utils/model.py†L380-L386】
- **Tokenization**: `model.tokenize(texts)` applies the tokenizer to a list of
  strings passed to it, applies chat template if it's present (for instruct models)
  and moves them to the model device. This yields `input_ids` and `attention_mask`.
  【F:src/lm_polygraph/utils/model.py†L703-L739】
- **Generation**: `model.generate(input_ids=..., **gen_args)` does:
  1. Merges default generation parameters (`self.generation_parameters`) with
     call‑site overrides, validates/removes unsupported HF `generate` args.
  2. Adds a `StoppingCriteriaList` (multi‑token EOS check) if `generate_until`
     is set.
  3. Wraps a custom `_ScoresProcessor` into `logits_processor` to capture
     original `log_softmax` scores before token processors are run by transformers.
  4. Calls `model.generate(**args)` and then swaps in the raw scores stored
     by `_ScoresProcessor` for downstream stat calculators.
  【F:src/lm_polygraph/utils/model.py†L510-L595】
- **Postprocessing**: `generate_texts(input_texts, **args)` tokenizes inputs,
  calls `generate` with `return_dict_in_generate=True`, strips prompt tokens,
  and decodes only the newly generated tokens (or entire sequences for seq2seq).
  【F:src/lm_polygraph/utils/model.py†L595-L630】

### 4.2 Deep Dive: Stat Calculator Registration

- **Configuration lookup and container creation**: `get_stat_calculator_names(config)` reads
  `config.stat_calculators` (a list of entries in the YAML), and for each:
  - If the entry is `auto`, it invokes `register_default_stat_calculators(...)`, which
    iterates through all built‑in `StatCalculator` classes, calls an internal `_register`
    helper to wrap each into a `StatCalculatorContainer` (capturing its name, builder
    function, default config via `OmegaConf`, dependencies, and stats produced).
  - For non‑auto entries, it directly wraps the YAML fields (`name`, `cfg`, `stats`,
    `dependencies`, `builder`) into `StatCalculatorContainer` instances.
  The result is a flat list of containers describing each calculator to run.
  【F:scripts/polygraph_eval†L170-L200】【F:src/lm_polygraph/defaults/register_default_stat_calculators.py†L1-L60】

**StatCalculatorContainer fields**:
```text
name         # The calculator's class name or YAML name
obj/builder  # Python callable or import path to build the StatCalculator instance
cfg          # OmegaConf config object with hyperparameters
stats        # List of statistic keys this calculator will produce
dependencies # List of other statistic keys required as inputs
```

### 4.3 Deep Dive: Estimator Instantiation

- `get_ue_methods(config, model)` uses `FactoryEstimator` to instantiate each
  estimator named in `config.estimators`, passing its `cfg` to the constructor.
  Each `Estimator` subclass then consumes batch stats to produce uncertainty scores.
  【F:scripts/polygraph_eval†L206-L213】

### 4.4 Deep Dive: Generation & UE Metrics

- `get_generation_metrics(args)` initializes ground-truth metrics (ROUGE, BLEU,
  BERTScore, Accuracy, AlignScore) and wraps them in preprocessing hooks
  if `process_output_fn`/`process_target_fn` is set.
  【F:scripts/polygraph_eval†L216-L289】
- `get_ue_metrics(args)` creates metrics for evaluating uncertainty estimation
  quality (PRAUC, ROCAUC, etc.), optionally adding claim-based UE metrics.
  【F:scripts/polygraph_eval†L157-L166】

### 4.5 Deep Dive: `UEManager` Core Workflow

- **Initialization** (`UEManager.init`):
  1. Collects all declared stat calculators (`StatCalculatorContainer`) into a dict mapping each
     statistic name to its container.
  2. Builds a dependency graph: for every calculator, records which `stats` it produces and which
     `dependencies` it requires.
  3. Aggregates the full list of needed statistic keys by combining:
     - `e.stats_dependencies` from each UE estimator `e`
     - `m.stats_dependencies` from each generation metric `m`
     - mandatory keys `greedy_texts` (and `greedy_tokens` for whitebox models)
  4. Calls `order_calculators(stats, stat_calculators_dict, stat_dependencies_dict)`, which
     topologically sorts the required calculators to satisfy dependencies and returns:
     - An ordered list of stat keys to compute, and
     - A set of all stats that will be available after running them.
  5. Filters out redundant entries (e.g. blackbox_ variants when underlying stats are already produced).
  6. Instantiates the selected `StatCalculator` instances via `FactoryStatCalculator`, passing
     each container’s builder and config.
  【F:src/lm_polygraph/utils/manager.py†L128-L200】
- **Per-batch `calculate`**: invokes each `StatCalculator` on `batch_stats`
  to augment statistics; failures can be ignored or raised.
  【F:src/lm_polygraph/utils/manager.py†L216-L287】
- **Per-batch `estimate`**: invokes each `Estimator(batch_stats)` to
  collect uncertainty scores in `self.estimations`.
  【F:src/lm_polygraph/utils/manager.py†L288-L350】
- **Execution loop** (`__call__`): iterates over dataset batches,
  calling `calculate` → `estimate` → ground-truth generation metrics →
  processors → final UE metric computations and correlation.
  【F:src/lm_polygraph/utils/manager.py†L350-L480】

## 5. Extending Stat Calculators and UE Estimators

### 5.1 Adding a Stat Calculator
1. Implement under `src/lm_polygraph/stat_calculators/`.
2. Reference it in your YAML:
```yaml
stat_calculators:
  - name: MyStat
    cfg:
      foo: bar
    stats: [s1, s2]
    dependencies: [OtherStat]
    builder: my_module.builder_fn
```

### 5.2 Adding a UE Estimator
1. Implement under `src/lm_polygraph/estimators/`.
2. Register via factory:
```python
@register_estimator("MyEstimator")
class MyEstimator(UEEstimatorBase):
    ...
```
3. Add to config:
```yaml
estimators:
  - name: MyEstimator
    cfg:
      param: value
```

## 6. Custom Metrics and Processing Hooks
- Customize generation metrics via `generation_metrics` or
  extend `src/lm_polygraph/generation_metrics/`.
- Use `process_output_fn` / `process_target_fn` to clean or filter text.

## 7. Putting It All Together
1. Create a new YAML under `examples/configs/` or
   `examples/configs/instruct/`, reusing defaults.
2. Implement any custom Python modules (calculators, estimators, hooks).
3. Run:
```bash
HYDRA_CONFIG=/path/to/your_config.yaml polygraph_eval
```

## 8. Practical Example: GSM8k

Below is an in-depth walkthrough of running the GSM8k benchmark using
`examples/configs/polygraph_eval_gsm8k.yaml`.

```yaml
# examples/configs/polygraph_eval_gsm8k.yaml
hydra:
  run:
    dir: ${cache_path}/${task}/${model}/${dataset}/${now:%Y-%m-%d}/${now:%H-%M-%S}

defaults:
  - model: bloomz-560m
  - estimators: default_estimators
  - stat_calculators: default_calculators
  - _self_

cache_path: ./workdir/output
save_path: '${hydra:run.dir}'
instruct: false
task: qa

dataset: ['LM-Polygraph/gsm8k', 'continuation']
text_column: input
label_column: output
train_split: train
eval_split: test
max_new_tokens: 256
load_from_disk: false
normalize: true
trust_remote_code: false
size: 10000
generation_params:
  generate_until:
    - "\n"

target_ignore_regex: "(?s).*#### "
output_ignore_regex: "(?s).*The answer is "

subsample_eval_dataset: -1
generation_metrics: null
ignore_exceptions: false
batch_size: 1
seed:
  - 1
```
【F:examples/configs/polygraph_eval_gsm8k.yaml†L1-L35】

### 7.1 Config composition

1. **Hydra defaults** pull in:
   - `model: bloomz-560m`, `estimators: default_estimators`,
     `stat_calculators: default_calculators`, and `_self_`
   【F:examples/configs/polygraph_eval_gsm8k.yaml†L4-L7】
2. **Dataset & generation settings** configure GSM8k, batch size,
   token limits, and text normalization via regex filters
   【F:examples/configs/polygraph_eval_gsm8k.yaml†L20-L30】
3. **Estimator presets** come from `default_estimators.yaml`:
   ```yaml
   - name: MaximumSequenceProbability
   - name: Perplexity
   - name: MeanTokenEntropy
   ...
   ```
   【F:examples/configs/estimators/default_estimators.yaml†L1-L30】
4. **Stat calculator presets** come from `default_calculators.yaml`:
   ```yaml
   - auto
   - name: TrainingStatisticExtractionCalculator
     builder: ...
   ```
   【F:examples/configs/stat_calculators/default_calculators.yaml†L1-L15】

### 7.2 Runtime pipeline

Run the benchmark:
```bash
HYDRA_CONFIG=examples/configs/polygraph_eval_gsm8k.yaml polygraph_eval
```

Inside `main(args)`, the order is:

```python
# 1) Build stat calculators first
stat_cals = get_stat_calculator_names(args)
```
【F:scripts/polygraph_eval†L170-L179】【F:scripts/polygraph_eval†L181-L189】

```python
# 2) Instantiate UE estimators
estimators = get_ue_methods(args, model)
```
【F:scripts/polygraph_eval†L207-L213】

```python
# 3) Initialize generation & UE metrics
generation_metrics = get_generation_metrics(args)
ue_metrics = get_ue_metrics(args)
```
【F:scripts/polygraph_eval†L216-L235】【F:scripts/polygraph_eval†L157-L166】

```python
# 4) Run UEManager (generation → stats → evaluation)
man = UEManager(
    data=dataset,
    model=model,
    estimators=estimators,
    builder_env_stat_calc=BuilderEnvironmentStatCalculator(model),
    available_stat_calculators=stat_cals,
    generation_metrics=generation_metrics,
    ue_metrics=ue_metrics,
    processors=[Logger()],
    ignore_exceptions=args.ignore_exceptions,
    max_new_tokens=args.max_new_tokens,
    log_time=getattr(args, "log_time", False),
)
man()
man.save(save_path + f"/ue_manager_seed{seed}")
```
【F:scripts/polygraph_eval†L127-L146】

### 7.3 End-to-end summary

- **Stat calculators** compute dataset-level statistics (e.g., training embeddings,
  likelihoods) _before_ uncertainty estimation.
- **Estimators** consume these statistics to produce uncertainty scores.
- **Generation metrics** (ROUGE/BLEU/BERTScore/Accuracy) and **UE metrics**
  (PRAUC/AUC) are computed over model outputs.
- Results and intermediate state are saved per-seed under
  `<save_path>/ue_manager_seed{seed}`.

_End of guide._
