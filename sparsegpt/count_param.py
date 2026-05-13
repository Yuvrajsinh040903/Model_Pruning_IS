import os
os.environ["HF_HOME"] = "/scratch/heli"
os.environ["HF_DATASETS_CACHE"] = "/scratch/heli" 

from transformers import AutoModelForCausalLM

base_model_name = "facebook/opt-1.3b"
pruned_model_name = "/scratch/heli/sparse_opt/opt-1.3B"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
pruned_model = AutoModelForCausalLM.from_pretrained(pruned_model_name, device_map="auto")

total_params = count_parameters(base_model)
print(f"Total: {total_params:,} ({total_params/1e6:.1f}M)")

total_params = count_parameters(pruned_model)
print(f"Total: {total_params:,} ({total_params/1e6:.1f}M)")

def sparsity_stats(model):
    total = 0
    nonzero = 0
    for param in model.parameters():
        total += param.numel()
        nonzero += (param != 0).sum().item()
    
    sparsity = 1 - (nonzero / total)
    print(f"Total params: {total:,}")
    print(f"Non-zero: {nonzero:,} ({nonzero/total*100:.1f}%)")
    print(f"Sparsity: {sparsity*100:.1f}%")

sparsity_stats(base_model)
sparsity_stats(pruned_model)
