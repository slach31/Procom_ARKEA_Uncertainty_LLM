from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import logging
import gc, re
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay, 
    classification_report
)

hf_token = "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"

os.environ["HF_TOKEN"] = hf_token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token


def sample_balanced(df, label_col, n_total=500, random_state=42):
    """Sample n_total rows evenly across label classes."""
    n_classes = df[label_col].nunique()
    n_per_class = n_total // n_classes
    df_balanced = (
        df.groupby(label_col, group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), n_per_class), random_state=random_state))
          .reset_index(drop=True)
          .sample(frac=1, random_state=random_state)
          .reset_index(drop=True)
    )
    print(f"Sampled {len(df_balanced)} rows ({n_per_class} per class).")
    return df_balanced


@torch.no_grad()
def ask_LLM_abt_dataset(
    model_name,
    query_qst,
    dataset,
    text_col="text",
    hf_token=None,
    top_k=10,
    max_new_tokens=20,
    temperature=0.3,
    nucleus_p=0.9,
    k_sampling=50,
    device=None,
):
    """Memory-safe LLM loop with EPR computation."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading {model_name} on {device}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    dataset = dataset.copy()
    dataset["EPR"] = 0.0
    dataset["LLM_response"] = ""
    dataset["output_length"] = 0

    try:
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_size = 2 if gpu_mem_total < 10 else 4 if gpu_mem_total < 20 else 6
    except Exception:
        batch_size = 1

    print(f"Using batch size = {batch_size}\n")

    for start in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset.iloc[start:start + batch_size]
        prompts = [query_qst + text + "\nAnswer:" for text in batch[text_col]]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=nucleus_p,
            top_k=k_sampling,
            return_dict_in_generate=True,
            output_scores=True,
        )

        seqs, scores = outputs.sequences, outputs.scores
        gen_len = len(scores)

        for i in range(len(batch)):
            input_len = (inputs["attention_mask"][i] == 1).sum().item()
            generated_ids = seqs[i][input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            generated_text = (
                generated_text.strip()
                               .lower()
                               .replace(".", "")
                               .replace("**", "")
                               .split()[0]
                if generated_text.strip()
                else "unknown"
            )

            total_ent = 0.0
            for score in scores:
                probs = F.softmax(score, dim=-1)
                top_probs, _ = torch.topk(probs, k=top_k)
                safe_probs = torch.clamp(top_probs, min=1e-9)
                ent = -(safe_probs * torch.log2(safe_probs)).sum(dim=-1)
                total_ent += ent[i].item()

            dataset.at[start + i, "EPR"] = total_ent / gen_len if gen_len > 0 else 0.0
            dataset.at[start + i, "LLM_response"] = generated_text
            dataset.at[start + i, "output_length"] = gen_len

        del inputs, outputs
        torch.cuda.empty_cache()
        if psutil.virtual_memory().percent > 85:
            gc.collect()

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"✅ Finished generating responses for {model_name}")
    return dataset


def save_llm_responses(df, text_col, label_col, dataset_name, model_name):
    """Save per-row results."""
    os.makedirs("outputs", exist_ok=True)
    df_export = df.rename(columns={text_col: "text", label_col: "label"})
    df_export["model"] = model_name
    df_export["dataset"] = dataset_name
    cols = [c for c in ["text", "label", "LLM_response", "EPR", "output_length", "model", "dataset"] if c in df_export.columns]
    filename = f"outputs/{model_name.replace('/', '_')}_{dataset_name}_responses.csv"
    df_export.to_csv(filename, columns=cols, index=False, encoding="utf-8")
    print(f"💾 Saved responses → {filename}")


def calculate_metrics(df, label_col, score_col="EPR", pred_col="LLM_response", model_name="model"):
    """
    Universal metric calculator — auto-detects binary vs multi-class.
    """
    os.makedirs("plots", exist_ok=True)
    os.makedirs("misclassified", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)  

    df = df.copy()
    unique_labels = sorted(df[label_col].dropna().unique())
    n_classes = len(unique_labels)

    # --- If binary classification ---
    if n_classes == 2:
        df[pred_col] = (
            df[pred_col]
            .astype(str)
            .str.lower()
            .apply(lambda x: re.findall(r"(toxic|non\-toxic|positive|negative)", x))
            .apply(lambda lst: lst[0] if lst else "unknown")
        )

        def safe_label_to_int(value):
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ["toxic", "positive"]:
                    return 1
                elif v in ["non-toxic", "negative"]:
                    return 0
            if isinstance(value, (int, float)):
                return int(value)
            return 0

        y_true = df[label_col].astype(int)
        y_pred = df[pred_col].apply(safe_label_to_int).astype(int)
        y_scores = df[score_col].astype(float)

        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        acc = accuracy_score(y_true, y_pred)
        correct_mask = (y_true == y_pred)
        mean_epr_correct = df.loc[correct_mask, score_col].mean()
        mean_epr_incorrect = df.loc[~correct_mask, score_col].mean()

        # --- Plots ---
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        prefix = f"plots/{model_name.replace('/', '_')}_{label_col}"
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {model_name} ({label_col})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{prefix}_ROC_curve.png", dpi=300)
        plt.close()

        plt.figure(figsize=(5,4))
        plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve — {model_name} ({label_col})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{prefix}_PR_curve.png", dpi=300)
        plt.close()

        print(f"\n--- {model_name.upper()} ---")
        print(f"ROC-AUC: {roc_auc:.3f} | PR-AUC: {pr_auc:.3f} | Accuracy: {acc*100:.2f}%")

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy": acc,
            "mean_epr_correct": mean_epr_correct,
            "mean_epr_incorrect": mean_epr_incorrect,
        }

    else:
        mapping = {"world": 1, "sports": 2, "business": 3, "sci/tech": 4, "science": 4, "tech": 4}
        df[pred_col] = df[pred_col].astype(str).str.lower().apply(lambda x: mapping.get(x, 0))
        y_true = df[label_col].astype(int)
        y_pred = df[pred_col].astype(int)

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # --- Save metrics ---
        metrics_path = f"metrics/{model_name.replace('/', '_')}_{label_col}_metrics.csv"
        report_df["accuracy_global"] = acc
        report_df.to_csv(metrics_path, index=True)
        print(f"📊 Saved multi-class metrics → {metrics_path}")

        # --- Confusion Matrix ---
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["World", "Sports", "Business", "Sci/Tech"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix — {model_name}")
        plt.tight_layout()
        plt.savefig(f"plots/{model_name.replace('/', '_')}_{label_col}_confusion_matrix.png", dpi=300)
        plt.close()

        # --- EPR by class ---
        plt.figure(figsize=(6,4))
        sns.barplot(x=y_true, y=df["EPR"])
        plt.title(f"Mean EPR per Class — {model_name}")
        plt.xlabel("Class ID")
        plt.ylabel("EPR")
        plt.tight_layout()
        plt.savefig(f"plots/{model_name.replace('/', '_')}_{label_col}_EPR_per_class.png", dpi=300)
        plt.close()

        print(f"✅ Multi-class accuracy: {acc*100:.2f}%")

                # --- Compute ROC-AUC (macro) ---
        from sklearn.preprocessing import label_binarize

        y_true_bin = label_binarize(y_true, classes=unique_labels)
        y_pred_bin = label_binarize(y_pred, classes=unique_labels)


        # --- Mean EPR per class ---
        mean_epr_per_class = df.groupby(label_col)["EPR"].mean().to_dict()

        # --- Append to metrics CSV ---
        extra_stats = {
            "mean_epr_global": df["EPR"].mean()
        }
        for cls, val in mean_epr_per_class.items():
            extra_stats[f"mean_epr_class_{cls}"] = val

        extra_stats_df = pd.DataFrame([extra_stats])
        extra_stats_df.to_csv(metrics_path, mode="a", index=False)

        print(f"Mean EPR (global): {df['EPR'].mean():.4f}")
        return