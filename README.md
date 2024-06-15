# Preparation
## **1. Packages Installation (Python 3.10 used)**
 - python.exe -m pip install --upgrade pip
 - pip3 install numpy
 - pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
 - pip3 install git+https://github.com/huggingface/accelerate.git -q -U
 - pip3 install git+https://github.com/huggingface/transformers.git -q -U
 - pip3 install peft
 - pip3 install scipy
 - pip3 install -U scikit-learn
 - pip3 install bitsandbytes

## **2. Code Changes**
 - Change: LLM/starcoder/finetune/run.py | (All) prepare_model_for_int8_training -> prepare_model_for_kbit_training
 - Change: LLM/starcoder/finetune/run.py | "/home/ma-user/modelarts/inputs/model_2/" -> "TheBloke/Wizard-Vicuna-13B-Uncensored-HF"
 - Add: vd_venv/lib/python3.10/site-packages/transformers/models/gpt_bigcode/configuration_gpt_bigcode.py/GPTBigCodeConfig | def set_special_params(self, args): self.args = vars(args)
 - Change: LLM/starcoder/finetune/run.py | sys.path.append("Specify your python location")

# Instructions
## **1. Request GPU (Change based on your demand)**
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
--learning_rate 5e-5 \
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
