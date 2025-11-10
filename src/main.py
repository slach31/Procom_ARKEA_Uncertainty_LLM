from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import *


model_google = "google/gemma-3-1b-it"
model_microsoft = "microsoft/Phi-4-mini-instruct"
#model_ministral = "ministral/Ministral-3b-instruct"

path_toxic = "toxic.csv"
path_movies = "IMDB_Dataset.csv"


# --- Load CSVs ---
df_toxic = pd.read_csv(path_toxic, on_bad_lines='skip', engine='python')
df_movies = pd.read_csv(path_movies, on_bad_lines='skip', engine='python')

# --- Preview ---
print("\n Toxic sample:")
print(df_toxic.head())

print("\n IMDb Movie Reviews sample:")
print(df_movies.head())


df_toxic = df_toxic.drop(['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1)
map_sentiment = {"positive" : 1, "negative" : 0}


df_toxic_balanced = sample_balanced(df_toxic, label_col='toxic', n_total=150)
df_movies_balanced = sample_balanced(df_movies, label_col='sentiment', n_total=150)
df_movies_balanced["sentiment"] = df_movies_balanced["sentiment"].map(map_sentiment)


# --- Preview ---
print("\n Toxic Balanced Sample:")
print(df_toxic_balanced.head())

print("\n IMDb Balanced Sample:")
print(df_movies_balanced.head())

query_qst_toxic = (
    "You are a text classification assistant.\n"
    "Classify the following text as 'toxic' or 'non-toxic'.\n"
    "Respond with exactly one word: toxic or non-toxic.\n\n"
    "Text:\n"
)


query_qst_movies = (
    "You are a sentiment classification assistant.\n"
    "Classify the following movie review as 'positive' or 'negative'.\n"
    "Respond with exactly one word: positive or negative.\n\n"
    "Review:\n"
)


df_toxic_google = ask_LLM_abt_dataset(
    model_name=model_google,
    query_qst=query_qst_toxic,
    dataset=df_toxic_balanced,
    text_col="comment_text"
)

df_movies_google = ask_LLM_abt_dataset(
    model_name=model_google,
    query_qst=query_qst_movies,
    dataset=df_movies_balanced,
    text_col="review"
)

df_toxic_microsoft = ask_LLM_abt_dataset(
    model_name=model_microsoft,
    query_qst=query_qst_toxic,
    dataset=df_toxic_balanced,
    text_col="comment_text"
)

df_movies_microsoft = ask_LLM_abt_dataset(
    model_name=model_microsoft,
    query_qst=query_qst_movies,
    dataset=df_movies_balanced,
    text_col="review"
)

"""
df_toxic_ministral = ask_LLM_abt_dataset(
    model_name=model_ministral,
    query_qst=query_qst_toxic,
    dataset=df_toxic_balanced,
    text_col="comment_text"
)

df_movies_ministral = ask_LLM_abt_dataset(
    model_name=model_ministral,
    query_qst=query_qst_movies,
    dataset=df_movies_balanced,
    text_col="review"
)
"""

df_movies_google.to_csv("df_movies_google.csv", index=False)
df_movies_microsoft.to_csv("df_movies_microsoft.csv", index=False)
#df_movies_ministral.to_csv("df_movies_ministral.csv", index=False)
df_toxic_google.to_csv("df_toxic_google.csv", index=False)
df_toxic_microsoft.to_csv("df_toxic_microsoft.csv", index=False)
#df_toxic_ministral.to_csv("df_toxic_ministral.csv", index=False)

calculate_metrics(df_toxic_google, label_col="toxic")
calculate_metrics(df_toxic_microsoft, label_col="toxic")
calculate_metrics(df_movies_google, label_col="sentiment")
calculate_metrics(df_movies_microsoft, label_col="sentiment")