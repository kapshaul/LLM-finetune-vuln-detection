# Preparation
## **1. Packages Installation (Python 3.10 used)**
 - pip install --upgrade pip
 - pip3 install numpy
 - pip3 install torch torchvision torchaudio
 - pip3 install accelerate
 - pip3 install transformers==4.34.1
 - pip3 install peft
 - pip3 install scipy

## **2. Code Changes**
 - Change: LLM/starcoder/finetune/run.py | (All) prepare_model_for_int8_training -> prepare_model_for_kbit_training
 - Change: LLM/starcoder/finetune/run.py | "/home/ma-user/modelarts/inputs/model_2/" -> "TheBloke/Wizard-Vicuna-13B-Uncensored-HF"
 - Add: vd_venv/lib/python3.10/site-packages/transformers/models/gpt_bigcode/configuration_gpt_bigcode.py/GPTBigCodeConfig | def set_special_params(self, args): self.args = vars(args)

# Instructions
## **1. Request GPU (Change based on your demand)**
srun -p dgxh --time=1-00:00:00 -c 2 --gres=gpu:4 --mem=40g --pty bash
 - Cluster: dgxh
 - Time: 1-00:00:00
 - #CPUs: 2
 - #GPUs: 4
 - Memory: 40g

## **2. Load openssl module**
module load openssl

## **3. Activate the virtual environment (This is customized for the code)**
source vd_venv/bin/activate

## **4. Use the below command to run (Remember to specify the correct path for the file)**
PYTHONPATH=/nfs/stak/users/leeyongh/study/school/AI-539/Project/vul-llm-finetune/LLM/starcoder python3 vul-llm-finetune/LLM/starcoder/finetune/run.py --dataset_tar_gz='vul-llm-finetune/Datasets/with_p3/java_k_1_strict_2023_06_30.tar.gz' --split="train" --seq_length 50 --batch_size 1 --gradient_accumulation_steps 160 --learning_rate 1e-4 --lr_scheduler_type="cosine" --num_warmup_steps 1 --weight_decay 0.05 --output_dir='vul-llm-finetune/outputs/results_0/' --log_freq=1 --delete_whitespaces --base_model starcoder --lora_r 8 --debug_on_small_model --several_funcs_in_batch
