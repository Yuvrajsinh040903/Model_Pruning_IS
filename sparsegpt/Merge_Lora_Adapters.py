import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys
from huggingface_hub import login

model_path = Path(sys.argv[1])
print("Model Mentioned is : ",model_path)
# Load the adapter config
tokenizer = AutoTokenizer.from_pretrained(model_path)

peft_config = PeftConfig.from_pretrained(model_path)
base_model_name = peft_config.base_model_name_or_path

print(f"Base model: {base_model_name}")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
base_model.resize_token_embeddings(len(tokenizer))

print("Loading the LoRA Adapters...")
# Load the LoRA adapter
model = PeftModel.from_pretrained(base_model, model_path)

# Merge the adapter with base model
print("Merging adapter with base model...")
merged_model = model.merge_and_unload()

# Save the merged model
output_dir = model_path.with_name(f"merged-{model_path.name}")
merged_model.save_pretrained(output_dir)

# Copy tokenizer from your fine-tuned directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(output_dir)

print(f"Merged model saved to: {output_dir}")
