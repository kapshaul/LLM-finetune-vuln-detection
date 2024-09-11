# Research Replication: LLM Fine-tuning for Code Vulnerability Detection

**Authors:** Yong-Hwan Lee, James Flora, Shijie Zhao, and Yunhan Qiao

## Overview

This project replicates and builds upon the study by *Shestov et al. (2024)*, aiming to validate and extend their findings. The original research focused on fine-tuning large language models (LLMs) for code vulnerability detection. The approach utilized `LoRA` (Low-Rank Adaptation), a technique that involves adding adapters within layers for fine-tuning. During this process, the original model parameters are *frozen*, and only the adapters are trained, making the training process more cost-effective.

A key innovation of our work is the incorporation of our custom adaptation of `QLoRA`, which first quantizes the LLM to a *4-bit float*, significantly reducing its size. For example, the **13B-WizardCoder model**, originally around *26 GB* and typically requiring more than *30 GB* of VRAM, is reduced to approximately *7 GB* after quantization. Following quantization, the `LoRA` technique is applied for fine-tuning.

### What is LoRA?

<div align="center">
    
<img src="https://github.com/kapshaul/llm-finetune-vuln-detection/blob/master/LoRA.png" width="500">

**Figure 1**: LoRA adapter illustration

</div>

Figure 1 illustrates how LoRA adapters can be significantly smaller than the original parameter sizes. The number of parameters for the $A$ adapter is $r \times k$, and for the $B$ adapter, it is $d \times r$. Considering the original parameter matrix is $d \times k$, where both $d$ and $k$ are usually large for LLMs, choosing a small $r$ can effectively reduce the number of parameters. Thus, the original matrix $W \in \mathbb{R}^{d \times k}$ is much larger than the combined size of the adapters $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$.

>For example, consider a layer in a LLM with a weight matrix $W \in \mathbb{R}^{1000 \times 100}$. The number of parameters for $W$ is $1000 \times 100 = 100,000$. If we set the LoRA rank to $r = 5$, the size of the LoRA adapters is only $1000 \times 5 + 100 \times 5 = 5,500$. This means the adapter size is around 5% of the original weight matrix $W$, which is significantly manageable for training as the original weight matrix $W$ remains frozen during the training phase.

<br>

In this project, we varied the `dataset`, `sequence length`, and `the use of focal loss`; measured the resulting performance changes compared to LoRA alone.
The report for this project: [PDF](https://github.com/kapshaul/llm-finetune-vuln-detection/blob/master/vuln_detection_finetune.pdf)

This document provides detailed instructions for replicating our research project. It includes steps for setting up the necessary environment, making required code changes, running the model on a High-Performance Computing (HPC) cluster, and presenting the results.

## Preparation
### **1. Packages Installation (Python 3.10 used)**
```bash
pip install -r requirements.txt
```

### **2. Code Change**
- For a debug model compatibility, Add the following function into the `GPTBigCodeConfig` class in the transformers package located at `your_venv/lib/python3.10/site-packages/transformers/models/gpt_bigcode/configuration_gpt_bigcode.py`:

```python
class GPTBigCodeConfig:
    # ... other methods and attributes ...

    def set_special_params(self, args):
        self.args = vars(args)
```

- Change the directory path at `./vul-llm-finetune/LLM/starcoder/run.py`
```python
sys.path.append("my_path/vul-llm-finetune/LLM/starcoder")
```

## Implementation Instruction
### **1. Request GPU from HPC (Based on OSU HPC server)**
srun -p dgxh --time=2-00:00:00 -c 2 --gres=gpu:2 --mem=20g --pty bash
 - Cluster: dgxh
 - Time: 2-00:00:00
 - #CPUs: 2
 - #GPUs: 2
 - Memory: 20g

### **2. Use the below command to run (Specify the path for model saving and loading)**
 - Debug using a small model
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

 - Train using LLM   
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

 - Test
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

## Result

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

## Conclusion

In this paper, we recreate the findings of *Shestov et al*. in which we finetune the LLM, WizardCoder, for code vulnerability detection. Whilst the original authors use LoRA  to do so, we employ QLoRA to cut down on overall model size and are able to train such a model on a consumer-grade GPU. Despite this, we see significant degradation in performance metrics though it is clear that the model is still doing some sort of *learning*. Further, we perform experimentation on the hyperparameters *sequence length* and *include large function*. We are able to conclude that including large functions is a strict positive for the model’s learning capabilities, but the evidence on sequence length is inconclusive due to a baffling experiment with much higher results than the rest.

## Reference

[1] Shestov, A., Levichev, R., Mussabayev, R., Maslov, E., Cheshkov, A., & Zadorozhny, P. (2024). *Finetuning Large Language Models for Vulnerability Detection*. arXiv preprint arXiv:2401.17010. Retrieved from [https://arxiv.org/abs/2401.17010](https://arxiv.org/abs/2401.17010).

[2] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685. Retrieved from https://arxiv.org/abs/2106.09685.

[3] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv preprint arXiv:2305.14314. Retrieved from https://arxiv.org/abs/2305.14314.
