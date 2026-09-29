# Reproduction Notes

This page covers what it takes to run `vul-llm-finetune/LLM/starcoder/finetune/run.py` from this snapshot. The commands below show how the code is *meant* to be invoked. None has been run against this snapshot, and none will run until the [known blockers](#known-blockers) are fixed and a compatible model is chosen. Nothing here establishes a working environment or a memory requirement. For project background, see the [README](../README.md).

## Known blockers

These issues are present in the checked-in code and dependency files.

1. **Indentation error (`run.py:386`).** The `print(f"Best model path found: ...")` line inside `run_test_peft` is indented by six spaces instead of eight. Python rejects the whole file with an `IndentationError` at compile time, so every mode fails, including `--help`.
2. **Placeholder import path (`run.py:29`).** `sys.path.append("my_path/vul-llm-finetune/LLM/starcoder")` is a literal placeholder. The imports that follow (`finetune.dataset`, `finetune.gpt_big_code_classification_several_funcs`, `utils.calc_quality`, and, indirectly, `utils.focal_loss`) need the `vul-llm-finetune/LLM/starcoder` directory on `sys.path`, for example by deriving the path from `__file__`. The `debug_funcs` import on line 33 resolves because Python puts the script's own directory (`finetune/`) on the path.
3. **Version-sensitive imports in `debug_funcs.py`.** Line 6 imports `transformers.deepspeed`, which later `transformers` releases deprecated in favor of `transformers.integrations`. `run.py` imports this module unconditionally, even though `debug_params` itself is never called.
4. **Unpinned, moving dependencies.** [`requirements.txt`](../requirements.txt) pins nothing, and `transformers` and `accelerate` install from their Git default branches, so each install can pull different code. The snapshot was written against 2023–2024 releases. Other version-sensitive spots:
   - `TrainingArguments(evaluation_strategy=...)` (`run.py:328`) is the older spelling of `eval_strategy`.
   - `load_in_4bit=True` is passed directly to `from_pretrained` (`run.py:262–266`). Newer releases expect a `BitsAndBytesConfig` passed as `quantization_config`.

### Other prerequisites

- **A compatible base model.** The config and model classes are hardcoded to GPTBigCode (StarCoder architecture). The `--LLM_path` default, `TheBloke/Wizard-Vicuna-13B-Uncensored-HF`, has not been verified to load with them. The report says the team used "WizardCoder-13B" from Hugging Face but does not give the exact model ID. Choose a checkpoint and confirm it loads with the GPTBigCode classes before training.
- **The quantized model cache.** No base model, quantized model, or trained adapter is included. `--load_quantized_model` reads from `--model_path`, which is empty in a fresh checkout. Without that flag, the script still loads `--LLM_path` in 4-bit, then saves the quantized model to `--model_path`.
- **Debug mode.** `--debug_on_small_model` avoids loading base-model weights, but it still downloads the tokenizer from `--LLM_path`. With `--several_funcs_in_batch`, it also needs an adaptation: `create_small_gptbigcode_config` (`run.py:209`) builds a stock `transformers.GPTBigCodeConfig`, yet line 270 calls `config.set_special_params(args)` and the classifier reads `config.args`. Only the local `GPTBigCodeConfigClassificationSeveralFunc` (`gpt_big_code_classification_several_funcs.py:27`) provides those. The historical workaround edited `GPTBigCodeConfig` inside the installed `transformers` package; building the debug config from the local class avoids that. The full-model path already uses the local class (`run.py:261`).

## Environment

- **Python.** The old README says Python 3.10 was used, and the committed `__pycache__` files are `cpython-310`.
- **Dependencies.** [`requirements.txt`](../requirements.txt) lists `numpy`, `torch`, `torchvision`, `torchaudio`, `accelerate` and `transformers` (from Git), `peft`, `scipy`, `scikit-learn`, and `bitsandbytes`. It is not a reliable one-shot install recipe: line 4 attaches `--index-url https://download.pytorch.org/whl/cu121` to `torchaudio`, which pip may not accept as a per-requirement option. `run.py` also imports `safetensors` and `tqdm`, which normally arrive as dependencies of `transformers`.
- **Other requirement files.** [`vul-llm-finetune/LLM/requirements.txt`](../vul-llm-finetune/LLM/requirements.txt) and [`llm.Dockerfile`](../vul-llm-finetune/LLM/llm.Dockerfile) belong to the upstream container setup. They include packages such as `guidance` and `click` for `next_token_prediction/`, and they omit `peft` and `bitsandbytes`.
- **Hardware.** The code assumes a CUDA GPU: loading uses 4-bit `bitsandbytes`, the device map is keyed on the accelerator process index, and adapter checkpoints are opened with `safe_open(..., device=0)`. The report ran on a Tesla T4, an RTX 4080, and 2× A100; the LoRA numbers it quotes from the original authors used 8× V100. These are the machines that happened to be used, not validated requirements.

## Key arguments

The upstream [LLM README](../vul-llm-finetune/LLM/README.md) lists every flag. The notes below cover the ones the report's runs depend on, plus behavior you can only see in the source.

| Argument | Meaning in this code |
| --- | --- |
| `--dataset_tar_gz` | Path to a dataset archive. `train`, `valid`, and `test` are always all loaded. |
| `--split` | Parsed but never used. The mode depends only on `--run_test` / `--run_test_peft`. |
| `--LLM_path` | Hugging Face ID or directory of the full-precision base model. Also the tokenizer source in every mode. |
| `--model_path` | Directory where the 4-bit model is saved, or loaded from when `--load_quantized_model` is set. Defaults to `./vul-llm-finetune/LLM/starcoder/quantized_model/`, relative to the working directory. |
| `--seq_length` | Token budget per packed sequence (default 2048; the report used 256 and 512). |
| `--ignore_large_functions` | Skips functions with at least `seq_length` tokens. Without it, those functions are truncated to `seq_length − 1` tokens, not kept whole. This is the "Large Function: ignore / include" column in the results. |
| `--several_funcs_in_batch` | Packs several functions into one sequence, separated by the EOS token, and classifies each function at its last token. The report's runs use it. |
| `--use_focal_loss`, `--focal_loss_gamma` | Focal loss in place of cross-entropy (gamma defaults to 2.0). |
| `--loss_reduction` | `sum` by default. The report found sum worked better than mean. |
| `--base_model` | `starcoder` (LoRA on `c_attn`, `c_proj`, `c_fc`, and the head `linear_layer`) or `codegen2`. |
| `--no_fp16` | Inverted (`store_false` with no explicit default): fp16 is **off** by default, and passing `--no_fp16` turns it **on**. |
| `--no_gradient_checkpointing` | Ineffective (`store_false` with `default=False`): the value is always `False`, so gradient checkpointing is always enabled. |

Avoid `--no_fp16` and `--no_gradient_checkpointing` unless you intend the behavior above or fix the argument definitions (`run.py:129–131`).

## Outputs

- `<output_dir>/checkpoint-<epoch>/`: the adapter with the best validation ROC AUC so far. It is replaced whenever a later epoch does better. It also contains an intentionally empty `adapter_model.bin` next to `adapter_model.safetensors`.
- `<output_dir>/final_checkpoint/`: saved after training, once the best-validation adapter has been reloaded. It therefore holds the best-epoch adapter, not the last-epoch one.
- `--model_path`: the saved 4-bit base model (only on runs without `--load_quantized_model`).
- Validation and test metrics (`roc_auc`, `f1`) are printed to standard output and the `Trainer` log. They are not written to a results file. The printed test F1 uses a threshold tuned on the test labels; see [Metric caveats](experiment-results.md#metric-caveats).

## Example commands (after repairs)

These are adapted from the team's original README. They assume blockers 1–4 are fixed and `--LLM_path` points to a GPTBigCode-compatible checkpoint you have chosen; add `--LLM_path=<model>` to each command, because the default is not a verified choice. The debug command also needs the debug-mode adaptation above. Run from the repository root. None has been run against this snapshot. `--split` is kept from the original commands but has no effect.

**Debug with a tiny randomly initialized GPTBigCode model.** This builds a 2-layer model, so no base-model weights are loaded; the tokenizer still comes from `--LLM_path`.

```bash
python vul-llm-finetune/LLM/starcoder/finetune/run.py \
--dataset_tar_gz='vul-llm-finetune/Datasets/with_p3/java_k_1_strict_2023_06_30.tar.gz' \
--split="train" \
--lora_r 8 \
--seq_length 512 \
--batch_size 1 \
--gradient_accumulation_steps 160 \
--learning_rate 1e-4 \
--weight_decay 0.05 \
--num_warmup_steps 2 \
--log_freq=1 \
--output_dir='vul-llm-finetune/outputs/results_test/' \
--delete_whitespaces \
--several_funcs_in_batch \
--debug_on_small_model
```

**Train with QLoRA.** This command uses `--load_quantized_model`, so it needs a quantized model already saved at `--model_path`. On a first run, drop that flag; the script then quantizes `--LLM_path` and saves it to `--model_path`.

```bash
python vul-llm-finetune/LLM/starcoder/finetune/run.py \
--dataset_tar_gz='vul-llm-finetune/Datasets/with_p3/java_k_1_strict_2023_06_30.tar.gz' \
--load_quantized_model \
--split="train" \
--lora_r 8 \
--use_focal_loss \
--focal_loss_gamma 1 \
--seq_length 512 \
--num_train_epochs 15 \
--batch_size 1 \
--gradient_accumulation_steps 160 \
--learning_rate 1e-4 \
--weight_decay 0.05 \
--num_warmup_steps 2 \
--log_freq=1 \
--output_dir='vul-llm-finetune/outputs/results_0/' \
--delete_whitespaces \
--base_model starcoder \
--several_funcs_in_batch
```

The table rows differ in `--dataset_tar_gz` (use `without_p3/java_k_1_strict_2023_07_03.tar.gz` for "X₁ without P₃"), `--seq_length`, and whether `--ignore_large_functions` is passed. The exact commands and seeds behind each historical row are not recorded.

**Evaluate a trained adapter on the test split.** Pass the same data and model flags that were used for training. `--run_test_peft`, not `--split`, selects this mode.

```bash
python vul-llm-finetune/LLM/starcoder/finetune/run.py \
--dataset_tar_gz='vul-llm-finetune/Datasets/with_p3/java_k_1_strict_2023_06_30.tar.gz' \
--load_quantized_model \
--split="test" \
--run_test_peft \
--lora_r 8 \
--seq_length 512 \
--checkpoint_dir='vul-llm-finetune/outputs/results_0' \
--model_checkpoint_path='final_checkpoint' \
--delete_whitespaces \
--base_model starcoder \
--several_funcs_in_batch
```

### Historical cluster allocation

The team requested GPUs on the Oregon State University HPC cluster with the following allocation. It records the team's own setup on that cluster, not a tested requirement:

```bash
srun -p dgxh --time=2-00:00:00 -c 2 --gres=gpu:2 --mem=20g --pty bash
```

That is partition `dgxh`, a two-day limit, 2 CPUs, 2 GPUs, and 20 GB of host memory.

## Other scripts

- `finetune/merge_peft_adapters.py` merges an adapter into a base model loaded with `AutoModelForCausalLM`. It was written for causal-LM adapters. It was not part of the report's classification workflow, and its `--push_to_hub` option uploads to the Hugging Face Hub.
- `next_token_prediction/llm_finetune_check.py`, `LineVul/`, and `ContraBERT/` are upstream baselines with their own dependencies and Docker files. The report does not use them.
