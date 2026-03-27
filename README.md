# Évaluation des LLM et expériences EPR

Ce dépôt contient plusieurs notebooks explorant l’évaluation par modèles de langage (LLM-as-a-Judge) ainsi que les méthodes EPR / WEPR sur différents jeux de données et modèles.

---

## Fichiers

### 1. `llm-judge-epr-compa-gemma.ipynb`
- Compare l’approche LLM-as-a-Judge avec les métriques EPR en utilisant le modèle Gemma.
- Objectif : analyser l’accord entre différentes méthodes d’évaluation.

---

### 2. `llm-judge-epr-compa-qwen.ipynb`
- Même comparaison que précédemment mais avec le modèle Qwen.
- Permet une analyse comparative entre modèles.

---

### 3. `llm-judge-gemma.ipynb`
- Implémente un pipeline d’évaluation LLM-as-a-Judge avec Gemma.
- Évalue les réponses générées selon leur qualité, pertinence ou exactitude.

---

### 4. `llm-judge-qwen.ipynb`
- Pipeline équivalent avec Qwen.
- Permet de comparer le comportement des modèles en tant qu’évaluateurs.

---

### 5. `epr+wepr-webquestions-qwen.ipynb`
- Applique les métriques EPR / WEPR sur le dataset WebQuestions avec Qwen.
- Focus sur l’évaluation de questions-réponses factuelles.

---

### 6. `epr-imdb-ministral.ipynb`
- Utilise EPR sur le dataset IMDB avec le modèle Ministral.
- Étude de l’évaluation sur des tâches de classification de texte (sentiment).

---

## Concepts clés

- LLM-as-a-Judge : utilisation d’un modèle de langage pour évaluer des réponses.
- EPR (Evntropy Production Rate)
- WEPR : version pondérée de EPR.
- Évaluation inter-modèles : comparaison du comportement de différents LLM.

---

## Objectifs

- Comparer LLM-as-a-Judge et les métriques EPR/WEPR
- Étudier la cohérence entre modèles (Gemma, Qwen, Ministral)
- Tester les méthodes sur différents datasets (WebQuestions, IMDB)

---

