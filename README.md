# Évaluation de l'Incertitude en Sortie des LLMs

Ce projet explore différentes méthodes pour quantifier et détecter l'incertitude dans les réponses générées par des grands modèles de langage (LLMs). Les expériences sont menées sur le dataset **WebQuestions** (questions factuelles en question-réponse) et portent sur trois approches complémentaires : **EPR**, **WEPR** et **LLM-as-a-Judge**.

## Méthodes Évaluées

### EPR — Entropy Production Rate

L'**EPR** (Entropy Production Rate) est une méthode de détection d'hallucinations en boîte noire qui exploite l'entropie produite au niveau des tokens lors de la génération [2]. L'idée centrale est que lorsqu'un modèle est incertain, la distribution de probabilité sur le vocabulaire est plus étalée, produisant une entropie plus élevée token par token.

Dans ce projet, l'EPR est calculé en générant plusieurs réponses pour chaque question (N=20 échantillons), en extrayant les scores de logits à chaque étape de décodage, puis en calculant l'entropie de Shannon sur la distribution des tokens. Une faible entropie moyenne signale une réponse confiante ; une entropie élevée indique une incertitude.

**Métriques clés :**
- Corrélation entropie / accuracy (Pearson & Spearman)
- Courbe ROC et AUC avec l'EPR comme score de confiance inversé
- Amélioration de l'accuracy par filtrage des réponses à haute entropie

### WEPR — Weighted Entropy Production Rate

Le **WEPR** est une variante pondérée de l'EPR [2]. Plutôt que de moyenner uniformément l'entropie sur tous les tokens générés, le WEPR attribue des poids différents aux tokens selon leur position ou leur importance dans la séquence. Cette pondération permet de mieux capturer les tokens sémantiquement critiques et de réduire le bruit introduit par des tokens fonctionnels (articles, ponctuation, etc.) à faible valeur informative.

### LLM-as-a-Judge

L'approche **LLM-as-a-Judge** [1] consiste à utiliser un second LLM (LLM2) pour évaluer la correction des réponses produites par un premier LLM (LLM1), sans accès aux labels de vérité terrain. LLM2 reçoit la question, la réponse de LLM1 et les réponses attendues, puis émet un jugement binaire (`CORRECT` / `INCORRECT`).

Cette méthode est évaluée en comparant les jugements de LLM2 avec la vérité terrain sur 100 questions WebQuestions.

**Métriques clés :**
- Accuracy réelle de LLM1 (vs vérité terrain)
- Accuracy du jugement de LLM2 (capacité à distinguer correct/incorrect)
- Accuracy perçue par LLM2 (biais d'optimisme ou de pessimisme)
- Matrice de confusion, précision, rappel, F1


## Structure du Projet

```
.
├── WebQuestions_EPR.ipynb          # Expériences EPR / WEPR sur WebQuestions
├── LLM_as_Judge_WebQuestions.ipynb   # Expériences LLM-as-a-Judge sur WebQuestions
└── README.md
```

### `WebQuestions_EPR.ipynb`

Ce notebook implémente l'analyse d'incertitude par entropie token-level sur le dataset WebQuestions avec le modèle **Mistral-7B-Instruct-v0.3** (et variantes testées : Qwen2.5-3B/7B, LLaMA-3.2-3B, Phi-4-mini, Gemma-2-9B).

Étapes principales :
1. Chargement du modèle via HuggingFace (accès GPU, `float16`)
2. Chargement des 1000 premières questions de `web_questions` (split test)
3. Génération de 20 réponses par question avec un prompt few-shot optimisé (température 0.1)
4. Calcul de l'entropie token-level à partir des `output_scores` de `model.generate()`
5. Agrégation par consensus (réponse la plus fréquente) et calcul de l'accuracy
6. Visualisations EPR : distribution des densités (correct vs incorrect), courbe ROC, accuracy par filtrage sur seuil d'entropie
7. Export des résultats en CSV

### `LLM_as_Judge_WebQuestions.ipynb`

Ce notebook implémente le pipeline LLM-as-a-Judge sur 100 questions WebQuestions avec des modèles HuggingFace locaux.

- **LLM1** : `google/flan-t5-xl` — modèle répondant aux questions
- **LLM2** : `google/flan-t5-large` — modèle juge évaluant les réponses

Étapes principales :
1. Chargement du dataset et sélection de 100 questions
2. Collecte des réponses de LLM1 via prompts factuels
3. Évaluation réelle par correspondance partielle normalisée avec la vérité terrain
4. Jugement de LLM2 (prompt demandant `CORRECT` ou `INCORRECT`)
5. Calcul des métriques : accuracy réelle, accuracy perçue, accuracy du jugement
6. Analyse du biais (FPR, FNR), matrice de confusion, rapport de classification
7. Visualisations : comparaison des accuracies, évolution cumulative, précision/rappel

## Dépendances

```bash
pip install datasets transformers torch accelerate \
            anthropic openai pandas numpy scipy \
            matplotlib seaborn scikit-learn tqdm
```

Un accès **HuggingFace Hub** avec token est requis pour les modèles sous licence (Mistral, LLaMA, Gemma).


## Références

[1] Haitao Li, Qian Dong, Junjie Chen, Huixue Su, Yujia Zhou, Qingyao Ai, Ziyi Ye, and Yiqun Liu. *LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods*, 2024.

[2] Charles Moslonka, Hicham Randrianarivo, Arthur Garnier, and Emmanuel Malherbe. *Learned Hallucination Detection in Black-Box LLMs Using Token-Level Entropy Production Rate*, 2026.
