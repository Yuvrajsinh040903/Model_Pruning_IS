import os

from vllm import LLM, SamplingParams
import torch
from datasets import load_from_disk
import pandas as pd
from pathlib import Path
import numpy as np
from huggingface_hub import login

models = ['Models/llama_1B_INST_FT_sparseGPT_pruned_unstructured_75',
 'Models/llama_1B_INST_FT_sparseGPT_pruned_structured_75_6_8',
 'Models/llama_1B_INST_trans_prompt_sparseGPT_pruned_structured_50_2_4',
 'Models/llama_3B_INST_sparseGPT_pruned_structured_25_2_8',
 'Models/llama_1B_INST_sparseGPT_pruned_structured_75_3_4',
 'Models/llama_3B_INST_sparseGPT_pruned_structured_25_1_4',
 'Models/llama_1B_INST_FT_sparseGPT_pruned_structured_25_2_8',
 'Models/llama_1B_INST_FT_sparseGPT_pruned_structured_75_3_4',
 'Models/llama_3B_INST_sparseGPT_pruned_structured_75_6_8',
 'Models/llama_1B_INST_FT_sparseGPT_pruned_unstructured_50',
 'Models/llama_1B_INST_FT_sparseGPT_pruned_structured_50_2_4',
 'Models/llama_1B_INST_trans_prompt_sparseGPT_pruned_structured_50_4_8',
 'Models/llama_1B_INST_trans_prompt_sparseGPT_pruned_structured_25_2_8',
 'Models/llama_1B_INST_FT_sparseGPT_pruned_unstructured_25',
 'Models/llama_1B_INST_FT_sparseGPT_pruned_structured_25_1_4',
 'Models/llama_1B_INST_trans_prompt_sparseGPT_pruned_unstructured_25',
 'Models/llama_1B_INST_trans_prompt_sparseGPT_pruned_unstructured_75',
 'Models/llama_3B_INST_sparseGPT_pruned_structured_50_4_8',
 'Models/llama_1B_INST_trans_prompt_sparseGPT_pruned_structured_75_3_4',
 'Models/llama_3B_INST_sparseGPT_pruned_unstructured_75',
 'Models/llama_1B_INST_FT_sparseGPT_pruned_structured_50_4_8',
 'Models/llama_1B_INST_trans_prompt_sparseGPT_pruned_structured_75_6_8',
 'Models/llama_3B_INST_sparseGPT_pruned_unstructured_25',
 'Models/llama_1B_INST_trans_prompt_sparseGPT_pruned_unstructured_50',
 'Models/llama_1B_INST_trans_prompt_sparseGPT_pruned_structured_25_1_4']

#### Variables ####
dataset_path = '../Layerwise/Data/Opus_Test_Data_en_hi'
model_path = models[24]
# "Qwen/Qwen3-4B", "Yuvrajsinh0409/Qwen_4B_LayerWise_Pruned_28_Layers", "meta-llama/Llama-3.2-1B-Instruct"
#### Load Dataset ####
data = load_from_disk(dataset_path)
# messages = data['prompt'][:10]

def format_chat(example):
    return {
        "messages":[{"role":"user","content":example['prompt']}]
    }
messages = data.map(format_chat)['messages'][:]
#### Load Model ####
llm = LLM(
    model=model_path,
    max_model_len=512,
    # dtype="auto",
    # gpu_memory_utilization=0.3,
    # trust_remote_code=True,
    tensor_parallel_size=1,
    enforce_eager=True,
    
)

#### Generate ####
outputs = llm.chat(
    messages,
    SamplingParams(max_tokens=256, temperature=0,stop=["</s>", "<|im_end|>", "<|endoftext|>"]),
    # extra_body={"chat_template_kwargs": {"enable_thinking": False}}  
    # default_chat_template_kwargs = {"enable_thinking": False}
    chat_template_kwargs={"enable_thinking": False}  

)

data = {
 "Source_Dialogue" : data['source'],
    "Reference": data['target'],
    "MT": [i.outputs[0].text.strip() for i in outputs]
}

infer_df = pd.DataFrame(data)
infer_df.to_csv(f"./Inference/{Path(model_path).name}.tsv", sep='\t')
