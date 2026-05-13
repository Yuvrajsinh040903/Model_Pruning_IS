from huggingface_hub import login, snapshot_download

snapshot_download(
    repo_id = "Yuvrajsinh0409/Llama_1B_inst_in22_flores_ft",
    repo_type = "model",
    local_dir = "./Llama_FT",
    cache_dir = "./Llama_FT"
        )
