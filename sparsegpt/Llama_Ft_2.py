import os
os.environ["HF_HOME"] = "/scratch/heli"
os.environ["HF_DATASETS_CACHE"] = "/scratch/heli"
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

model_name = "meta-llama/Llama-3.2-1B-Instruct"
output_dir = "/scratch/heli/Llama_1B_inst_in22_flores_ft"
train_file = "./train_dataset.tsv"

# -----------------------
# TOKENIZER (MANDATORY)
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------
# DATASET
# -----------------------
dataset = load_dataset(
    "csv",
    data_files=train_file,
    delimiter="\t",
    split="train"
)

if "Unnamed: 0" in dataset.column_names:
    dataset = dataset.remove_columns("Unnamed: 0")

def preprocess_function(example):
    user = example["prompt"].split("Target:")[0].strip()
    assistant = example["prompt"].split("Target:")[1].strip()
    return {
        "messages": [{"role": "user", "content": user},{"role": "assistant", "content": assistant}],
    }

dataset = dataset.map(preprocess_function, remove_columns=["prompt"])
dataset = dataset.shuffle(seed=42)
# -----------------------
# QLORA CONFIG
# -----------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# -----------------------
# MODEL
# -----------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.config.use_cache = False

# -----------------------
# LORA CONFIG
# -----------------------


peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_rslora=True, # Using Rank Stabilized LoRA
    task_type="CAUSAL_LM",
    
    # To target all layers :
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    
    # To target language model specific layers only :
    # target_modules = r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj$" ,
    
    
    ###Optimization###
    #use_dora=True,
    init_lora_weights="gaussian",
    #loftq_config=dict(loftq_bits=4),
    modules_to_save=["lm_head"],                      
)
#model = prepare_model_for_kbit_training(model)
#model = get_peft_model(model, peft_config)
#model.print_trainable_parameters()

# -----------------------
# SFT CONFIG
# -----------------------
sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    logging_steps=1,
    save_steps=100,
    bf16=True,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    report_to="none",
    save_strategy="steps",
    # save_total_limit=5,

    # completion_only_loss=True,
    assistant_only_loss=True,
    # max_length=4096,

    dataset_text_field="messages",
)

# -----------------------
# TRAINER
# -----------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
    args=sft_config,
)

trainer.train()
trainer.save_model(output_dir)
