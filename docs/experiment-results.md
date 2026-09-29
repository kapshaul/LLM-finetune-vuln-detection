# Experiment Results

These are the results reported in the [project report](../vuln_detection_finetune.pdf) (Table 1, Figures 1–2), from the team's 2024 runs. They are historical and unreproduced: the repository contains no logs, checkpoints, or saved predictions from those runs, and the numbers have not been regenerated from this snapshot. Read them together with the [metric caveats](#metric-caveats). For how to set up a new run, see [reproduction.md](reproduction.md).

## Setup

- **Task.** Binary classification of Java functions as vulnerable or not vulnerable.
- **Model.** Per the report, WizardCoder: the team chose "WizardCoder-13B" from Hugging Face because the original paper doesn't name its exact model version. The exact model ID is not recorded, and it differs from the code's `--LLM_path` default. The model was loaded in 4-bit and fine-tuned with LoRA for 15 epochs, with function batch-packing and focal loss.
- **Datasets.** Both datasets come from the original study, which drew on CVEfixes, a manually curated NVD dataset, and VCMatch. Counts below are from the checked-in archives (train / valid / test):
  - **X₁ with P₃** (`Datasets/with_p3/`) is the raw, heavily imbalanced set: 13,247 / 5,131 / 4,576, 22,954 in total, with 646 vulnerable functions (about 1 per 34 non-vulnerable). The report states 22,945 samples, which appears to be a transposition of the archive count.
  - **X₁ without P₃** (`Datasets/without_p3/`) is a much smaller, roughly balanced set: 810 / 272 / 252, 1,334 in total, with 665 vulnerable functions.
- **Varied factors.** Dataset, sequence length (256 or 512 tokens), and whether functions longer than the sequence length were skipped ("ignore") or truncated to fit ("include").

## Reported results

|          | Dataset       | Sequence Length | Large Function | ROC AUC | F1 Score | GPU            | Training Time (hr) |
|:--------:|:-------------:|:---------------:|:--------------:|:-------:|:--------:|:--------------:|:------------------:|
| **QLoRA**| X₁ without P₃ |       512       |     ignore     |  0.53   |   0.65   |    Tesla T4     |        8.2         |
|          | X₁ without P₃ |       512       |    include     |  0.56   |   0.66   | NVIDIA A100 x2  |        3.4         |
|          | X₁ without P₃ |       256       |     ignore     |  0.51   |   0.63   |    Tesla T4     |        2.9         |
|          | X₁ with P₃    |       512       |     ignore     |  0.68   |   0.14   |    RTX 4080     |       22.1         |
|          | X₁ with P₃    |       512       |    include     |  0.72   |   0.17   | NVIDIA A100 x2  |       20.4         |
|          | X₁ with P₃    |       256       |     ignore     |  0.70   |   0.14   | NVIDIA A100 x2  |       18.3         |
| **LoRA** | X₁ without P₃ |      2048       |    include     |  0.69   |   0.71   | NVIDIA V100 x8  |                    |
|          | X₁ with P₃    |      2048       |    include     |  0.86   |   0.27   | NVIDIA V100 x8  |                    |

Notes on the table:

- The **LoRA rows are Shestov et al.'s published numbers**, quoted from their paper. The report says the team could not run LoRA with its computing resources. The original paper doesn't report training time, so those cells are blank; the PDF shows "?".
- The RTX 4080 row appears as "GTX 4080" in the PDF table. The report text calls the card a consumer-grade NVIDIA RTX 4080.
- The report does not say how many runs or seeds each QLoRA row represents. The rows also differ in GPU type and count, so the training-time column mixes hardware and does not isolate the effect of sequence length or long-function handling.

## Figures

The per-epoch curves exist only in the PDF. No separate image files or source logs are included.

- **Figure 1:** validation ROC AUC and F1 per epoch for the four sequence-length runs (with/without P₃ × 256/512).
- **Figure 2:** validation loss per epoch for the same four runs.

The report observes that the smaller "without P₃" runs have much more volatile validation loss, and that validation metrics don't closely track the steadily falling loss.

## Findings

Each item gives the report's finding and how far the recorded evidence supports it.

- **QLoRA vs. LoRA.** The report finds QLoRA lower on both datasets, with F1 "dropping by some 10% on average" and lower ROC AUC. The LoRA side is quoted, not rerun, so this is not a controlled QLoRA-vs-LoRA ablation: quantization, sequence length (256/512 vs. 2048), hardware, and the team running the experiment all change at once.
- **Long functions.** At 512 tokens, the "include" run scored higher than the "ignore" run on both datasets (ROC AUC 0.56 vs. 0.53 and 0.72 vs. 0.68; F1 0.66 vs. 0.65 and 0.17 vs. 0.14). That is two paired comparisons. Each pair also used different GPUs, "include" means truncation rather than full-length functions, repetitions are unstated, and no significance test was done. The rows are consistent with a benefit but don't establish one in general.
- **Sequence length.** The evidence was inconclusive. The "with P₃" run at 256 tokens started with much higher validation ROC AUC, F1, and loss than the 512-token "with P₃" run. On the test set, though, it ended only 0.02 higher in ROC AUC and no higher in F1. The report could not explain this outlier.
- **Imbalance.** F1 is reported for the positive (vulnerable) class only. The low F1 on "with P₃" reflects its roughly 1:34 class imbalance.

The report's conclusion reads: *"We are able to conclude that including large functions is a strict positive for the model's learning capabilities, but the evidence on sequence length is inconclusive due to a baffling experiment with much higher results than the rest."* Given the two confounded comparisons above, the recorded data supports a weaker reading: an association in these runs, not a strict or causal benefit.

## Metric caveats

These follow from the evaluation code in `run.py` (`EvalQuality`) and `utils/calc_quality.py`:

- **F1 threshold is tuned on the labels being scored.** `EvalQuality` calls `quality_full_report_val(probs, labels)`, which passes the same predictions and labels as both the "validation" and "test" inputs of `quality_full_report`. The decision threshold that maximizes positive-class F1 is therefore chosen on the set being evaluated. This happens for per-epoch validation and also for `trainer.predict(test_data)`, so the printed test F1 is a best-threshold F1 on the test labels, not F1 at a threshold fixed in advance on validation data. If the reported F1 values came from this code path, as the snapshot suggests, they are optimistic and not directly comparable with F1 at a pre-set threshold. ROC AUC does not depend on a threshold.
- **Only `roc_auc` and `f1` are produced.** In `quality_full_report`, the `return` statement sits inside the metric loop (`calc_quality.py:88`), so the `macro_f1` and `macro_recall` metrics described in the module docstring are never reached.

## Implementation note

Section 3.1 of the report says the adapters target the attention layers only. In the code, `--base_model starcoder` targets `c_attn`, `c_proj`, `c_fc`, and the classification head `linear_layer` (`run.py:296`). In GPTBigCode, `c_fc` and the MLP's `c_proj` belong to the feed-forward block, so as written the code also adapts the MLP layers.
