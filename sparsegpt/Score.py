# ============================================================
# MT Evaluation Script
# Metrics:
# 1. COMET (Unbabel/wmt22-comet-da)
# 2. BLEU
# 3. chrF++
# 4. BERTScore
# 5. Prism
#
# Input TSV columns:
# Source_Dialogue | Reference | MT
#
# ============================================================

# =========================
# INSTALL REQUIREMENTS
# =========================
# Run these commands first:

# pip install pandas sacrebleu bert-score unbabel-comet torch sentencepiece
# pip install git+https://github.com/facebookresearch/fairseq.git
# pip install prismmt

# NOTE:
# Prism requires extra model download automatically.
# COMET model will also download automatically.

# ============================================================

import pandas as pd
import sacrebleu
from bert_score import score as bertscore
from comet import download_model, load_from_checkpoint
# from prism import Prism
import torch

# ============================================================
# LOAD TSV
# ============================================================

# Change this path
tsv_file = "./Inference/Inference_vllm_Llama_3.2_1B_Base.tsv"

df = pd.read_csv(tsv_file, sep="\t")

# Remove unnamed index columns if present
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print(df.head())

# ============================================================
# PREPARE DATA
# ============================================================

sources = df["Source_Dialogue"].astype(str).tolist()
references = df["Reference"].astype(str).tolist()
hypotheses = df["MT"].astype(str).tolist()

# ============================================================
# 1. BLEU
# ============================================================

bleu = sacrebleu.corpus_bleu(
    hypotheses,
    [references]
)

print("\n================ BLEU ================")
print("BLEU Score:", bleu.score)

# ============================================================
# 2. chrF++
# ============================================================

# word_order=2 => chrF++
chrf = sacrebleu.corpus_chrf(
    hypotheses,
    [references],
    word_order=2
)

print("\n================ chrF++ ================")
print("chrF++ Score:", chrf.score)

# ============================================================
# 3. BERTScore
# ============================================================

# Hindi model
P, R, F1 = bertscore(
    hypotheses,
    references,
    lang="hi",
    verbose=True
)

bertscore_avg = F1.mean().item()

print("\n================ BERTScore ================")
print("Average BERTScore F1:", bertscore_avg)

# Store sentence-level scores
df["BERTScore_F1"] = [x.item() for x in F1]

# ============================================================
# 4. COMET
# ============================================================

print("\nDownloading COMET model...")

model_path = download_model("Unbabel/wmt22-comet-da")

comet_model = load_from_checkpoint(model_path)

# COMET input format
comet_data = [
    {
        "src": str(src),
        "mt": str(mt),
        "ref": str(ref)
    }
    for src, mt, ref in zip(sources, hypotheses, references)
]

comet_scores = comet_model.predict(
    comet_data,
    batch_size=8,
    gpus=1 if torch.cuda.is_available() else 0
)

sentence_comet_scores = comet_scores["scores"]
system_comet_score = comet_scores["system_score"]

print("\n================ COMET ================")
print("System COMET Score:", system_comet_score)

df["COMET"] = sentence_comet_scores

# ============================================================
# 5. COMET
# ============================================================

print("\nDownloading COMTAIL model...")

def comtail_da(sources, predictions, references):
    # path to downloaded model's checkpoint
    da_model_path = '../COMTAIL-DA/checkpoints/epoch=4-step=230265.ckpt'
    da = load_from_checkpoint(da_model_path)
    data = {"src": [str(i) for i in sources], "mt": [str(i) for i in predictions], "ref": [str(i) for i in references]}
    data = [dict(zip(data, t)) for t in zip(*data.values())]
    da_output = da.predict(data, gpus=1, progress_bar=True)
    return da_output

da_scores = comtail_da(sources, hypotheses, references)
sentence_comtail_scores = da_scores.scores
system_comtail_score = da_scores.system_score
# print("COMTAIL-DA Scores:", da_scores)
df['COMTAIL'] = sentence_comtail_scores




# ============================================================
# 5. Prism
# ============================================================

# print("\nLoading Prism model...")

# # Automatically downloads model first time
# prism = Prism(
#     model_dir=None,
#     lang="hi"
# )

# # Sentence-level Prism scores
# prism_scores = prism.score(
#     cand=hypotheses,
#     ref=references,
#     segment_scores=True
# )

# # Average score
# avg_prism = sum(prism_scores) / len(prism_scores)

# print("\n================ Prism ================")
# print("Average Prism Score:", avg_prism)

# df["Prism"] = prism_scores

# ============================================================
# SAVE RESULTS
# ============================================================

output_file = "mt_evaluation_results.tsv"

df.to_csv(
    output_file,
    sep="\t",
    index=False
)

print("\n================================================")
print("Evaluation completed successfully!")
print("Results saved to:", output_file)
print("================================================")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n================ FINAL SCORES ================")

print(f"BLEU       : {bleu.score:.4f}")
print(f"chrF++     : {chrf.score:.4f}")
print(f"BERTScore  : {bertscore_avg:.4f}")
print(f"COMET      : {system_comet_score:.4f}")
print(f"COMTAIL      : {system_comtail_score:.4f}")
