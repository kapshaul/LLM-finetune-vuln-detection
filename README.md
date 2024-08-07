# Research Replication
This document provides detailed instructions for replicating our research project. The steps include setting up the necessary environment, making required code changes, and running the model on a High-Performance Computing (HPC) cluster.

The report for this project: [Link to PDF](https://github.com/neurokimchi/llm-finetune-vuln-detection/blob/vuln_detection_finetune.pdf)


# Preparation
## **1. Packages Installation (Python 3.10 used)**
 - pip install -r requirements.txt

## **2. Code Changes**
 - Add: your_venv/lib/python3.10/site-packages/transformers/models/gpt_bigcode/configuration_gpt_bigcode.py/GPTBigCodeConfig | def set_special_params(self, args): self.args = vars(args)

# Implementation Instructions
## **1. Request GPU from HPC (Change based on your demand)**
srun -p dgxh --time=2-00:00:00 -c 2 --gres=gpu:2 --mem=20g --pty bash
 - Cluster: dgxh
 - Time: 2-00:00:00
 - #CPUs: 2
 - #GPUs: 2
 - Memory: 20g

## **2. Use the below command to run (Specify the path for model saving and loading)**
 - Debug using a small model
```
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
```
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
```
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
