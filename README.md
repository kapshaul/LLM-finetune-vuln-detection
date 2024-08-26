# Research Replication: LLM Fine-tuning for Code Vulnerability Detection

**Authors:** Yong-Hwan Lee, James Flora, Shijie Zhao, and Yunhan Qiao

## Overview

The original research utilized LoRA, a technique that involves adding adapters within layers for fine-tuning. During this process, the original model parameters are `frozen`, and only the adapters are trained, making the training process more cost-effective. In our project, we adopted QLoRA, which first quantizes the large language model (LLM) to a `4-bit float` to reduce its size, making it more manageable. After quantization, it applies the LoRA technique.

<div align="center">
    
<img src="https://github.com/kapshaul/llm-finetune-vuln-detection/blob/VD/LoRA.png" width="500">

**Figure 1**: LoRA adapter illustration

</div>

Figure 1 illustrates how LoRA adapters can be significantly smaller than the original parameter sizes. The number of parameters for the $A$ adapter is $r \times k$, and for the $B$ adapter, it is $d \times r$. Considering the original parameter matrix is $d \times k$, where both $d$ and $k$ are usually large for LLMs, choosing a small $r$ can effectively reduce the number of parameters. Thus, the original matrix $W \in \mathbb{R}^{d \times k}$ is much larger than the combined size of the adapters $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$.

For example, consider a layer in a LLM with a weight matrix $W \in \mathbb{R}^{1000 \times 100}$. The number of parameters for $W$ is $1000 \times 100 = 100,000$. If we set the LoRA rank to $r = 5$, the size of the LoRA adapters is only $1000 \times 5 + 100 \times 5 = 5,500$. This means the adapter size is around 5% of the original weight matrix, which is significantly manageable for training.

<br>

<br>

In this project, we varied the `dataset`, `sequence length`, and `the use of focal loss`; measured the resulting performance changes compared to LoRA alone.
The report for this project: [PDF](https://github.com/kapshaul/llm-finetune-vuln-detection/blob/VD/vuln_detection_finetune.pdf)

This document provides detailed instructions for replicating our research project. It includes steps for setting up the necessary environment, making required code changes, running the model on a High-Performance Computing (HPC) cluster, and presenting the results.

## Preparation
### **1. Packages Installation (Python 3.10 used)**
```bash
pip install -r requirements.txt
```

### **2. Code Changes**
For a debug model compatibility, add the following function to the transformers package at `your_venv/lib/python3.10/site-packages/transformers/models/gpt_bigcode/configuration_gpt_bigcode.py`:

```python
class GPTBigCodeConfig:
    # ... other methods and attributes ...
    
    def set_special_params(self, args):
        self.args = vars(args)
```

## Implementation Instructions
### **1. Request GPU from HPC (Change based on your demand)**
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
|          | X₁ with P₃    |       512       |     ignore     |  0.68   |   0.14   |    GTX 4080     |       22.1         |
|          | X₁ with P₃    |       512       |    include     |  0.72   |   0.17   | NVIDIA A100 x2  |       20.4         |
|          | X₁ with P₃    |       256       |     ignore     |  0.70   |   0.14   | NVIDIA A100 x2  |       18.3         |
| **LoRA** | X₁ without P₃ |      2048       |    include     |  0.69   |   0.71   | NVIDIA V100 x8  |         ?          |
|          | X₁ with P₃    |      2048       |    include     |  0.86   |   0.27   | NVIDIA V100 x8  |         ?          |
