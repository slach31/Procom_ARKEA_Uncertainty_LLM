from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score


hf_token = "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 "

def sample_balanced(df, label_col, n_total=500, random_state=42):
    """
    Sample n_total rows evenly across label classes.
    """

    n_classes = df[label_col].nunique()
    n_per_class = n_total // n_classes

    # Sample evenly
    df_balanced = (
        df.groupby(label_col, group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), n_per_class), random_state=random_state))
          .reset_index(drop=True)
          .sample(frac=1, random_state=random_state)
          .reset_index(drop=True)
    )

    print(f"Sampled {len(df_balanced)} rows ({n_per_class} per class).")
    return df_balanced

df_toxic_balanced = sample_balanced(df_toxic, label_col='toxic', n_total=150)
df_movies_balanced = sample_balanced(df_movies, label_col='sentiment', n_total=150)
df_movies_balanced["sentiment"] = df_movies_balanced["sentiment"].map(map_sentiment)

def ask_LLM_abt_dataset(model_name, query_qst, dataset, text_col="review",
                        hf_token=hf_token, top_k=20, max_new_tokens=15):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token).to(device)
    model.eval()

    dataset = dataset.copy()
    dataset["EPR"] = 0.0
    dataset["LLM_response"] = ""
    dataset["output_length"] = 0

    for i in range(len(dataset)):
        prompt = query_qst + dataset[text_col].iloc[i] + "\nAnswer:"
        inputs = tokenizer(
              prompt,
              return_tensors="pt",
              truncation=True,
              max_length=2048,  # avoids warning
              padding=False
        ).to(device)


        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True
        )

        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Clean and normalize response
        generated_text = (
            generated_text.strip()
                           .lower()
                           .replace(".", "")
                           .replace("**", "")
                           .split()[0] if generated_text.strip() else "unknown"
        )

        total_ent = 0.0
        for score in outputs.scores:
            probs = F.softmax(score, dim=-1)
            top_probs, _ = torch.topk(probs, k=top_k)
            safe_probs = torch.clamp(top_probs, min=1e-9)
            ent = -(safe_probs * torch.log2(safe_probs)).sum().item()
            total_ent += ent

        if len(outputs.scores) > 0:
            dataset.at[i, "EPR"] = total_ent / len(outputs.scores)
            dataset.at[i, "output_length"] = len(outputs.scores)

        dataset.at[i, "LLM_response"] = generated_text

        if i % 10 == 0:
            print(f"Processed {i+1}/{len(dataset)}")

    return dataset

def calculate_metrics(df, label_col = "toxic") :

  # true labels (0 = non-toxic, 1 = toxic)
  y_true = df[label_col]

  # model scores (EPR values)
  y_scores = df["EPR"]

  # ROC-AUC (area under the ROC curve)
  roc_auc = roc_auc_score(y_true, y_scores)

  # PR-AUC (area under the Precision-Recall curve)
  pr_auc = average_precision_score(y_true, y_scores)

  print(f"ROC-AUC: {roc_auc:.4f}")
  print(f"PR-AUC:  {pr_auc:.4f}")
