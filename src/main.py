from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score
)
from utils import *
import warnings
import psutil
import logging


logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

model_google = "google/gemma-3-1b-it"
model_microsoft = "microsoft/phi-1_5"
model_ministral = "ministral/Ministral-3b-instruct"

path_toxic = "toxic.csv"
path_movies = "IMDB_Dataset.csv"
path_agnews = "mult.csv"


# --- Load CSVs ---
df_toxic = pd.read_csv(path_toxic, on_bad_lines='skip', engine='python')
df_movies = pd.read_csv(path_movies, on_bad_lines='skip', engine='python')
df_agnews = pd.read_csv(path_agnews, on_bad_lines='skip', engine='python')

# --- Preview ---
print("\n Toxic sample:")
print(df_toxic.head())

print("\n IMDb Movie Reviews sample:")
print(df_movies.head())

print("\n AG News sample:")
print(df_agnews.head())

df_toxic = df_toxic.drop(['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1)
map_sentiment = {"positive" : 1, "negative" : 0}


df_toxic_balanced = sample_balanced(df_toxic, label_col='toxic', n_total=150)
df_movies_balanced = sample_balanced(df_movies, label_col='sentiment', n_total=150)
df_movies_balanced["sentiment"] = df_movies_balanced["sentiment"].map(map_sentiment)
df_agnews_balanced = sample_balanced(df_agnews, label_col='Class Index', n_total=200)



# --- Preview ---
print("\n Toxic Balanced Sample:")
print(df_toxic_balanced.head())

print("\n IMDb Balanced Sample:")
print(df_movies_balanced.head())

print("\n AG News Balanced Sample:")
print(df_agnews_balanced.head())

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

query_qst_agnews = (
    "You are a text classification assistant.\n"
    "Your task is to read a short news article and classify it into one of the following categories:\n"
    "1. World\n2. Sports\n3. Business\n4. Science/Technology\n"
    "Respond with exactly one of these choices : World, Sports, Business, or Sci/Tech.\n\n"
    "Article:\n"
)

"""
df_toxic_google = ask_LLM_abt_dataset(
    model_name=model_google,
    query_qst=query_qst_toxic,
    dataset=df_toxic_balanced,
    text_col="comment_text"
)
"""


"""
df_movies_google = ask_LLM_abt_dataset(
    model_name=model_google,
    query_qst=query_qst_movies,
    dataset=df_movies_balanced,
    text_col="review"
)
"""

"""
df_agnews_google = ask_LLM_abt_dataset(
    model_name=model_google,
    query_qst=query_qst_agnews,
    dataset=df_agnews_balanced,
    text_col="Description"
)   
"""

"""
df_toxic_microsoft = ask_LLM_abt_dataset(
    model_name=model_microsoft,
    query_qst=query_qst_toxic,
    dataset=df_toxic_balanced,
    text_col="comment_text"
)
"""
"""
df_movies_microsoft = ask_LLM_abt_dataset(
    model_name=model_microsoft,
    query_qst=query_qst_movies,
    dataset=df_movies_balanced,
    text_col="review"
)
"""

"""
df_agnews_microsoft = ask_LLM_abt_dataset(
    model_name=model_microsoft,
    query_qst=query_qst_agnews,
    dataset=df_agnews_balanced,     
    text_col="Description"
)
"""



"""
df_toxic_ministral = ask_LLM_abt_dataset(
    model_name=model_ministral,
    query_qst=query_qst_toxic,
    dataset=df_toxic_balanced,
    text_col="comment_text"
)
"""

"""
df_movies_ministral = ask_LLM_abt_dataset(
    model_name=model_ministral,
    query_qst=query_qst_movies,
    dataset=df_movies_balanced,
    text_col="review"
)
"""


df_agnews_ministral = ask_LLM_abt_dataset(
    model_name=model_ministral,
    query_qst=query_qst_agnews,
    dataset=df_agnews_balanced,
    text_col="Description"
)


# --- Google Gemma ---
#save_llm_responses(df_toxic_google, text_col="comment_text", label_col="toxic", dataset_name="toxic", model_name="google/gemma-3-1b-it")

#save_llm_responses(df_movies_google, text_col="review", label_col="sentiment", dataset_name="movies", model_name="google/gemma-3-1b-it")
  

# --- Microsoft Phi ---
#save_llm_responses(df_toxic_microsoft, text_col="comment_text", label_col="toxic", dataset_name="toxic", model_name="microsoft/phi-4-mini-instruct")

#save_llm_responses(df_movies_microsoft, text_col="review", label_col="sentiment", dataset_name="movies", model_name="microsoft/phi-4-mini-instruct")

# --- Ministral 3B ---
#save_llm_responses(df_toxic_ministral, text_col="comment_text", label_col="toxic", dataset_name="toxic", model_name="ministral/Ministral-3b-instruct")

#save_llm_responses(df_movies_ministral, text_col="review", label_col="sentiment", dataset_name="movies", model_name="ministral/Ministral-3b-instruct") 

# --- AG News ---
#save_llm_responses(df_agnews_google, text_col="Description", label_col="Class Index", dataset_name="agnews", model_name="google/gemma-3-1b-it")
#save_llm_responses(df_agnews_microsoft, text_col="Description", label_col="Class Index", dataset_name="agnews", model_name="microsoft/phi-4-mini-instruct")  
save_llm_responses(df_agnews_ministral, text_col="Description", label_col="Class Index", dataset_name="agnews", model_name="ministral/Ministral-3b-instruct")  

results = []

#results.append(["google/gemma-3-1b-it", "toxic", calculate_metrics(df_toxic_google, "toxic", model_name="google_gemma")])
#results.append(["google/gemma-3-1b-it", "movies", calculate_metrics(df_movies_google, "sentiment", model_name="google_gemma")])
#results.append(["microsoft/Phi-1_5", "toxic", calculate_metrics(df_toxic_microsoft, "toxic", model_name="phi_1_5")])
#results.append(["microsoft/Phi-1_5", "movies", calculate_metrics(df_movies_microsoft, "sentiment", model_name="phi_1_5")])
#results.append(["ministral/Ministral-3b-instruct", "toxic", calculate_metrics(df_toxic_ministral, "toxic", model_name="ministral_3b")])
#results.append(["ministral/Ministral-3b-instruct", "movies", calculate_metrics(df_movies_ministral, "sentiment", model_name="ministral_3b")])
#results.append(["google/gemma-3-1b-it", "agnews", calculate_metrics(df_agnews_google, "Class Index", model_name="google_gemma")])
#results.append(["microsoft/Phi-1_5", "agnews", calculate_metrics(df_agnews_microsoft, "Class Index", model_name="phi_1_5")])
results.append(["ministral/Ministral-3b-instruct", "agnews", calculate_metrics(df_agnews_ministral, "Class Index", model_name="ministral_3b")])    


rows = []
for model, dataset, metrics in results:
    rows.append({
        "model": model,
        "dataset": dataset,
        "ROC-AUC": metrics["roc_auc"],
        "PR-AUC": metrics["pr_auc"],
        "Accuracy": metrics["accuracy"],
        "Mean EPR (Correct)": metrics["mean_epr_correct"],
        "Mean EPR (Incorrect)": metrics["mean_epr_incorrect"]
    })

results_df = pd.DataFrame(rows)
results_df.to_csv("Final_Results.csv", index=False)
print("\n Saved full results to Final_Results.csv")
