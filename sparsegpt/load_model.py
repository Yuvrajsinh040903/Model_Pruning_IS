import os
os.environ["HF_HOME"] = "/scratch/heli"
os.environ["HF_DATASETS_CACHE"] = "/scratch/heli" 

from transformers import AutoModelForCausalLM

#model_name = "meta-llama/Llama-3.2-1B-Instruct"
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", local_dir="/scratch/heli/models_from_hf/llama_3b_inst")

from datasets import load_dataset
#data = load_dataset('ptb-text-only/ptb_text_only', 'penn_treebank', split='train')
#print(data[0])  # Should show {'sentence': '...'}
# Try using the v1 version which is usually more stable on HF mirrors
#['wikitext-103-v1', 'wikitext-2-v1', 'wikitext-103-raw-v1', 'wikitext-2-raw-v1']
traindata = load_dataset('Salesforce/wikitext',split='train')
testdata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1' ,split='test')
print("Done!!")
