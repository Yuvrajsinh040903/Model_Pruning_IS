import os
os.environ["HF_HOME"] = "/scratch/heli"
os.environ["HF_DATASETS_CACHE"] = "/scratch/heli"
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
from statistics import mean
import polars as pl
from torch.cuda import empty_cache
from torch import nn
from tqdm.auto import tqdm
import sacrebleu
import gc
import logging
from datasets import load_dataset, load_from_disk
from sacrebleu.metrics import BLEU, CHRF
# from comet import download_model, load_from_checkpoint
from transformers.utils import logging as tf_logging


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# --- METRIC INITIALIZATION ---
SELECTED_METRIC = "bleu"
print(f"Initializing metric: {SELECTED_METRIC}")

chrf_metric = None
bleu_metric = None
comet_model = None

if SELECTED_METRIC == "chrf":
    chrf_metric = CHRF(word_order=2)
elif SELECTED_METRIC == "bleu":
    bleu_metric = BLEU(effective_order=True)
elif SELECTED_METRIC == "comet":
    from comet import download_model, load_from_checkpoint
    print("Loading COMET model... (this may take a moment)")
    comet_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                torch_dtype=torch.bfloat16,
                                                 # device_map=device
                                                ).to(device).eval()

    assert model.device.type == "cuda"

    print("\nOriginal model loaded:\n", model_name)

    model_parameters = count_parameters(model)
    print(f"\nModel parameters\n: {model_parameters:,}")

    print("\nOriginal number of layers\n:",
        model.model.config.num_hidden_layers
        )

    return model

model = load_model(model_name)
total_layers = model.model.config.num_hidden_layers

ds = load_from_disk("./Data/openlanguagedata/flores_plus/", "default")
# in22_gen = load_from_disk("ai4bharat/IN22-Gen")
# in22_conv = load_from_disk("ai4bharat/IN22-Conv")


tgt_langs = ds["devtest"]["iso_639_3"]
src_langs = ds["devtest"]["iso_639_3"]

eng_ds = ds['devtest'].filter(lambda x: x['iso_639_3']=='eng')[:500]
hin_ds = ds['devtest'].filter(lambda x: x['iso_639_3']=='hin')[:500]


eng_hin = []
hin_eng = []

eng_src = []
hin_src = []

# prompt = """Translate the following text from {SOURCE_LANGUAGE} to {TARGET_LANGUAGE}.
# Ensure the translation is accurate, fluent, and faithful to the original meaning.
# Do not add, remove, or change any information.

# Text:
# {SOURCE_TEXT}"""

prompt = """Translate following {SOURCE_LANGUAGE} dialogue to {TARGET_LANGUAGE}.\nSource Sentence:\n{SOURCE_TEXT}"""

eng_src.extend([*eng_ds['text']])#, *in22_gen['test']['eng_Latn'], *in22_conv['test']['eng_Latn']])
hin_src.extend([*hin_ds['text']])#, *in22_gen['test']['hin_Deva'], *in22_conv['test']['hin_Deva']])

eng_hin.extend([prompt.format(SOURCE_LANGUAGE="English", TARGET_LANGUAGE="Hindi", SOURCE_TEXT=s.strip()) for s in eng_ds['text']])
# eng_hin.extend([prompt.format(SOURCE_LANGUAGE="English", TARGET_LANGUAGE="Hindi", SOURCE_TEXT=s.strip()) for s in in22_gen['test']['eng_Latn']])
# eng_hin.extend([prompt.format(SOURCE_LANGUAGE="English", TARGET_LANGUAGE="Hindi", SOURCE_TEXT=s.strip()) for s in in22_conv['test']['eng_Latn']])

hin_eng.extend([prompt.format(SOURCE_LANGUAGE="Hindi", TARGET_LANGUAGE="English", SOURCE_TEXT=s.strip()) for s in hin_ds['text']])
# hin_eng.extend([prompt.format(SOURCE_LANGUAGE="Hindi", TARGET_LANGUAGE="English", SOURCE_TEXT=s.strip()) for s in in22_gen['test']['hin_Deva']])
# hin_eng.extend([prompt.format(SOURCE_LANGUAGE="Hindi", TARGET_LANGUAGE="English", SOURCE_TEXT=s.strip()) for s in in22_conv['test']['hin_Deva']])

source_sentences = eng_src
references = hin_src
# PROMPTS
####################################
prompts = eng_hin
####################################

def define_max_len(tgt_lang):
    lens = [len(sent.split())
            for sent, lang in zip(source_sentences, tgt_langs)
            if lang == tgt_lang]
    max_len = int(mean(lens) * 3)
    return max_len

max_len = 1024#define_max_len(tgt_lang="hin")

print(max_len)

# max_len = define_max_len(tgt_lang="eng")

# print(max_len)


def translate(prompts, model):

    batch_size = 20  # If memory does not allow, it should be smaller.
    print("Batch Size:", batch_size)

    translations = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]

        # Format all messages in the batch
        batch_messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]

        # Tokenize the entire batch and get attention mask
        batch_inputs = tokenizer.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True , # This returns both input_ids and attention_mask
            enable_thinking=False 
        )

        input_ids = batch_inputs['input_ids'].to(device)
        attention_mask = batch_inputs['attention_mask'].to(device)

        # Store original lengths for each sequence in the batch
        original_length = input_ids.shape[1]  # All sequences have same length due to padding

        # Generate for the entire batch with attention mask
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids,
                attention_mask=attention_mask,  # Pass the attention mask
                max_new_tokens=max_len,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        # Decode batch results
        for j, tokens in enumerate(gen_tokens):
            # Get the length of the original input for this specific sequence
            new_tokens = tokens[original_length:]
            translation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            translations.append(translation)

    return translations
True


all_dfs = []

total_remove = 3  # Change if needed
print("Total layers to remove:", total_remove)

layers_to_remove = [3]
ignore = set([] + layers_to_remove)
full = set(range(total_layers))


for i in tqdm(range(total_remove), desc="Downscaling", total=total_remove):

    print(f"\n---Run {len(layers_to_remove)} - Layers removed: {layers_to_remove}---")

    main_df = pl.DataFrame({
        "Layer": [],
        "Score": []
    }).cast({"Layer": pl.Int64, "Score": pl.Float64})

    best_score = float('-inf')
    best_layer = None

    required = sorted(list(full - ignore))

    for n in tqdm(required, total=len(required), position=1, desc="Layers eval"):

        print("\nPruning layer", n, "\n")

        # Clear any existing model
        if 'model' in locals():
            del model

        model = None
        gc.collect()
        with torch.no_grad():
            empty_cache()
        
        # Load the model
        model = load_model(model_name)

        layers_to_keep = list(full - set(layers_to_remove + [n]))

        lm_layers = model.model.layers

        # Update the number of layers
        model.model.layers = nn.ModuleList([lm_layers[n] for n in layers_to_keep])

        # Ensure the config reflects the actual number of layers
        model.config.num_hidden_layers = len(model.model.layers)

        ################################################
        for index, layer in enumerate(model.model.layers):
            # Update the index in the attention module (where the error originates)
            if hasattr(layer, "self_attn"):
                layer.self_attn.layer_idx = index
            
            # Update the index on the layer itself (good practice for compatibility)
            if hasattr(layer, "layer_idx"):
                layer.layer_idx = index
        ################################################
        
        # Check the new number of parameters
        model_parameters = count_parameters(model)
        print(f"New model parameters: {model_parameters:,}")

        # Check the new number of layers
        print("New number of layers:",
              model.model.config.num_hidden_layers,
             )


        # Translation

        translations = translate(prompts, model)

        # Evaluation

        current_score = 0.0

        if SELECTED_METRIC == "chrf":
            # sacrebleu expects [[ref1, ref2]] for references
            current_score = chrf_metric.corpus_score(translations, [references]).score
            
        elif SELECTED_METRIC == "bleu":
            current_score = bleu_metric.corpus_score(translations, [references]).score
            
        elif SELECTED_METRIC == "comet":
            # COMET expects list of dicts
            comet_data = [{"src": s, "mt": t, "ref": r} 
                          for s, t, r in zip(source_sentences, translations, references)]
            # Adjust batch_size based on your GPU VRAM
            comet_output = comet_model.predict(comet_data, batch_size=8, gpus=1)
            current_score = comet_output.system_score

        # Round for display cleanliness
        if SELECTED_METRIC != "comet":
            current_score = round(current_score, 2)

        print(f"Layer {n} -> {SELECTED_METRIC.upper()}: {current_score}")

        if current_score > best_score:
            best_score = current_score
            best_layer = n

        print("Best layer to remove so far:", best_layer, "Score:", best_score)

        df = pl.DataFrame({"Layer": n,
                           "Score": current_score,
                          }
                         )

        main_df = main_df.vstack(df)

        print(df)

        del translations
        gc.collect()

    layers_to_remove.append(best_layer)
    ignore.add(best_layer)
    print("\nLayers to remove:", layers_to_remove)


    all_dfs.append(main_df)
    main_df.write_ndjson(f"json/df_{i}.ndjson")

    with pl.Config(tbl_rows=total_layers):
        print(main_df)

with pl.Config(tbl_rows=total_layers):
    for layer_df in all_dfs:
        print(layer_df)

print("End of evaluation. Layers to remove:", layers_to_remove)
