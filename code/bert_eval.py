import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# ========= PATHS =========
PROJECT_ROOT = Path(__file__).resolve().parent.parent  
DATA_DIR = PROJECT_ROOT / "data" / "processed_data"

BERT_OUTPUT_DIR = PROJECT_ROOT / "bert_output"
DIR_BERT_RANDOM = BERT_OUTPUT_DIR / "random_split"
DIR_BERT_TIME   = BERT_OUTPUT_DIR / "time_split"
DIR_BERT_SUM    = BERT_OUTPUT_DIR / "summary"

for d in [DIR_BERT_RANDOM, DIR_BERT_TIME, DIR_BERT_SUM]:
    d.mkdir(parents=True, exist_ok=True)

# ========= CONFIG =========
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 3
SEED = 42
torch.manual_seed(SEED)


# ========= CONFUSION MATRIX SAVER =========
def save_confusion_matrix(cm, tag: str, model_name: str, split: str):
    if split == "random":
        out_dir = DIR_BERT_RANDOM
    elif split == "time":
        out_dir = DIR_BERT_TIME
    else:
        out_dir = BERT_OUTPUT_DIR / "other"
        out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix [{tag} | {model_name}]")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ham(0)", "Spam(1)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Ham(0)", "Spam(1)"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    out_path = out_dir / f"cm_{tag}_{model_name}.png"
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"🖼️  Saved confusion matrix: {out_path}")


# ========= TRAIN + EVALUATE ONE DATASET =========
def train_and_evaluate(df, name, split="random", tag=None):
    if tag is None:
        tag = f"{name}_{split}"

    print(f"\n=== DistilBERT on [{name}] ({split} split) ===")

    # ----- Split -----
    if split == "time" and "timestamp" in df.columns and df["timestamp"].notna().any():
        df = df.sort_values("timestamp")
        n = len(df)
        n_train = max(1, int(n * 0.8))
        train_df = df.iloc[:n_train]
        test_df = df.iloc[n_train:]
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)

    # ----- Tokenization -----
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text_clean"], padding="max_length", truncation=True, max_length=MAX_LEN)

    train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True).rename_column("label", "labels")
    test_ds  = Dataset.from_pandas(test_df).map(tokenize, batched=True).rename_column("label", "labels")

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # ----- Model & Trainer -----
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir=f"bert_output/{name}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=SEED,
        fp16 = torch.cuda.is_available(),
        save_strategy = "no",
        logging_strategy = "no"
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds)
    trainer.train()

    # ----- Evaluation -----
    preds = trainer.predict(test_ds)
    preds_labels = np.argmax(preds.predictions, axis=1)
    true_labels = preds.label_ids

    acc = accuracy_score(true_labels, preds_labels)
    prec, recall, f1, _ = precision_recall_fscore_support(true_labels, preds_labels, average="binary", zero_division=0)

    print(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    # ----- Confusion Matrix -----
    cm = confusion_matrix(true_labels, preds_labels, labels=[0, 1])
    save_confusion_matrix(cm, tag=tag, model_name="DistilBERT", split=split)

    return {
        "model": "DistilBERT",
        "tag": tag,
        "dataset": name,
        "split": split,
        "acc": acc,
        "prec": prec,
        "recall": recall,
        "f1": f1,
    }


# ========= MAIN PIPELINE =========
def main():
    datasets = ["spam_assassin", "enron", "trec2007", "combined"]
    results = []

    for name in datasets:
        path = DATA_DIR / f"{name}_clean.csv"
        if not path.exists():
            print(f"❌ Skipping {name}, file not found.")
            continue

        df = pd.read_csv(path).dropna(subset=["text_clean", "label"])
        df = df[df["text_clean"].str.len() > 0]
        df["label"] = df["label"].astype(int)

        # ----- Random Split -----
        res_rand = train_and_evaluate(df, name=name, split="random", tag=f"{name}_rand")
        results.append(res_rand)

        # ----- Time Split (if timestamps exist) -----
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            res_time = train_and_evaluate(df, name=name, split="time", tag=f"{name}_time")
            results.append(res_time)
        else:
            print(f"[{name}] No timestamp column or all missing — skipping time split.")

    # ----- Save Summary -----
    df_results = pd.DataFrame(results)
    out_path = DIR_BERT_SUM / "baseline_metrics_bert.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\n✅ Saved DistilBERT results summary: {out_path}")


# ========= ENTRY POINT =========
if __name__ == "__main__":
    main()