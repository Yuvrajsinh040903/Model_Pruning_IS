# ============================================================
# MULTI-FILE MT EVALUATION SCRIPT
# ============================================================

# ============================================================
# INSTALL REQUIREMENTS
# ============================================================

# pip install pandas sacrebleu bert-score unbabel-comet torch sentencepiece
# pip install git+https://github.com/facebookresearch/fairseq.git
# pip install prismmt

# ============================================================

import os
import glob
import pandas as pd
import sacrebleu
import torch

from bert_score import score as bertscore
from comet import download_model, load_from_checkpoint

# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing TSV files
INPUT_FOLDER = "./Infer"

# Folder to save sentence-level outputs
OUTPUT_FOLDER = "./sentence_scores"

# Final system-level summary file
FINAL_OUTPUT_FILE = "final_system_scores.tsv"

# COMTAIL checkpoint
COMTAIL_MODEL_PATH = "../COMTAIL-DA/checkpoints/epoch=4-step=230265.ckpt"

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# LOAD COMET MODEL ONLY ONCE
# ============================================================

print("\nLoading COMET model...")

comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

# ============================================================
# LOAD COMTAIL MODEL ONLY ONCE
# ============================================================

print("\nLoading COMTAIL model...")

comtail_model = load_from_checkpoint(COMTAIL_MODEL_PATH)

# ============================================================
# STORE FINAL SYSTEM RESULTS
# ============================================================

final_results = []

# ============================================================
# GET ALL TSV FILES
# ============================================================

tsv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.tsv"))

print(f"\nFound {len(tsv_files)} TSV files.")

# ============================================================
# LOOP THROUGH FILES
# ============================================================

for tsv_file in tsv_files:

    print("\n====================================================")
    print(f"Processing File: {tsv_file}")
    print("====================================================")

    try:

        # ====================================================
        # LOAD FILE
        # ====================================================

        df = pd.read_csv(tsv_file, sep="\t")

        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df = df.fillna('-')
        print("Df Ready!")
        # ====================================================
        # PREPARE DATA
        # ====================================================

        sources = df["Source_Dialogue"].astype(str).tolist()
        references = df["Reference"].astype(str).tolist()
        hypotheses = df["MT"].astype(str).tolist()
        print("Extrcted src, mt, ref")
        # ====================================================
        # BLEU
        # ====================================================
        print("BLUE")
        bleu = sacrebleu.corpus_bleu(
            [str(i) for i in hypotheses],
            [[str(i) for i in references]]
        )

        bleu_score = bleu.score

        # ====================================================
        # chrF++
        # ====================================================
        print("chrf++")
        chrf = sacrebleu.corpus_chrf(
            hypotheses,
            [references],
            word_order=2
        )

        chrf_score = chrf.score

        # ====================================================
        # BERTScore
        # ====================================================
        print("BERTScore")
        P, R, F1 = bertscore(
            hypotheses,
            references,
            lang="hi",
            verbose=False
        )

        bert_f1_scores = [x.item() for x in F1]
        bert_f1_avg = F1.mean().item()

        df["BERTScore_F1"] = bert_f1_scores

        # ====================================================
        # COMET
        # ====================================================
        print("COMET")
        comet_data = [
            {
                "src": str(src),
                "mt": str(mt),
                "ref": str(ref)
            }
            for src, mt, ref in zip(
                sources,
                hypotheses,
                references
            )
        ]

        comet_output = comet_model.predict(
            comet_data,
            batch_size=64,
            gpus=1 if torch.cuda.is_available() else 0
        )

        sentence_comet_scores = comet_output["scores"]
        system_comet_score = comet_output["system_score"]

        df["COMET"] = sentence_comet_scores

        # ====================================================
        # COMTAIL
        # ====================================================
        print("COMTAIL")
        comtail_data = [
            {
                "src": str(src),
                "mt": str(mt),
                "ref": str(ref)
            }
            for src, mt, ref in zip(
                sources,
                hypotheses,
                references
            )
        ]

        comtail_output = comtail_model.predict(
            comtail_data,
            batch_size=64,
            gpus=1 if torch.cuda.is_available() else 0
        )

        sentence_comtail_scores = comtail_output["scores"]
        system_comtail_score = comtail_output["system_score"]

        df["COMTAIL"] = sentence_comtail_scores

        # ====================================================
        # SAVE SENTENCE-LEVEL FILE
        # ====================================================

        base_name = os.path.basename(tsv_file)

        sentence_output_path = os.path.join(
            OUTPUT_FOLDER,
            base_name
        )

        df.to_csv(
            sentence_output_path,
            sep="\t",
            index=False
        )

        print(f"\nSaved sentence scores: {sentence_output_path}")

        # ====================================================
        # STORE SYSTEM-LEVEL RESULTS
        # ====================================================

        final_results.append({
            "File_Name": base_name,
            "BLEU": round(bleu_score, 4),
            "chrF++": round(chrf_score, 4),
            "BERTScore_F1": round(bert_f1_avg, 4),
            "COMET": round(system_comet_score, 4),
            "COMTAIL": round(system_comtail_score, 4)
        })

        # ====================================================
        # PRINT SCORES
        # ====================================================

        print("\n============== SCORES ==============")

        print(f"BLEU            : {bleu_score:.4f}")
        print(f"chrF++          : {chrf_score:.4f}")
        print(f"BERTScore_F1    : {bert_f1_avg:.4f}")
        print(f"COMET           : {system_comet_score:.4f}")
        print(f"COMTAIL         : {system_comtail_score:.4f}")

    except Exception as e:

        print(f"\nError processing {tsv_file}")
        print(str(e))

# ============================================================
# SAVE FINAL SYSTEM SUMMARY
# ============================================================

final_df = pd.DataFrame(final_results)

final_df.to_csv(
    FINAL_OUTPUT_FILE,
    sep="\t",
    index=False
)

# ============================================================
# DONE
# ============================================================

print("\n====================================================")
print("ALL FILES PROCESSED SUCCESSFULLY")
print("====================================================")

print(f"\nSentence-level outputs folder: {OUTPUT_FOLDER}")
print(f"System-level summary file: {FINAL_OUTPUT_FILE}")