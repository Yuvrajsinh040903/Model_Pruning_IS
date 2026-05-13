from huggingface_hub import login, snapshot_download
from pathlib import Path

repo_ids = [
    "Yuvrajsinh0409/llama_3B_INST_sparseGPT_pruned_unstructured_75",
    "Yuvrajsinh0409/llama_3B_INST_sparseGPT_pruned_unstructured_25",
    "Yuvrajsinh0409/llama_3B_INST_sparseGPT_pruned_structured_50_4_8",
    "Yuvrajsinh0409/llama_3B_INST_sparseGPT_pruned_structured_25_2_8",
    "Yuvrajsinh0409/llama_3B_INST_sparseGPT_pruned_structured_25_1_4",
    # "Yuvrajsinh0409/llama_3B_INST_sparseGPT_pruned_structured" #2_4
]



for repo_id in repo_ids:
    r = Path(repo_id)
    snapshot_download(
    repo_id = repo_id,
    repo_type = "model",
    local_dir = f"./Models/{r.name}",
    # cache_dir = f"./Models/cache/{r.name}"
        )
