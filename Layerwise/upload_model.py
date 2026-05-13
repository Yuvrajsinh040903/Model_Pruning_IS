from huggingface_hub import HfApi,login
from pathlib import Path
from tqdm.auto import tqdm
from itertools import chain

api = HfApi()

base_model = Path('/scratch/heli')
# model_dirs = [i/"Fine_Tuned_Models" for i in base_model.glob("*") if i.is_dir()]
# model_dirs.remove(Path('/nlsasfs/home/dibd/dibd-ilmt/iiithyd/priyda/Indo_Dialogues/MODELS/Mutual_Dialogues/Fine_Tuned_Models'))
models = ['llama_3B_INST_sparseGPT_pruned_structured_25_1_4',
 'llama_3B_INST_sparseGPT_pruned_structured_25_2_8',
 'llama_3B_INST_sparseGPT_pruned_structured_50_4_8',
 'llama_3B_INST_sparseGPT_pruned_unstructured_25',
 'llama_3B_INST_sparseGPT_pruned_unstructured_75']

# model_dirs = [i for i in base_model.glob("*")]
model_dirs = [Path(base_model/i) for i in models]

fine_tuned_models = model_dirs #list(chain.from_iterable(i.glob("*") for i in model_dirs))

def upload_model(name, folder_path):
    api.upload_large_folder(
            repo_id=f"Yuvrajsinh0409/{name}",
            repo_type="model",
            folder_path=folder_path
            )
    print("#"*50)
    print(f"{name} uploaded successfully!!")
    print("#"*50)


def push_models_to_hub(fine_tuned_models):
    for model in fine_tuned_models:
        name = model.name
        upload_model(name, folder_path=model)

if __name__ == "__main__":
    push_models_to_hub(fine_tuned_models)