# 🧠 EESA: Ensemble-Based Explainable Sentiment Analysis

> A hybrid NLP/ML/LLM framework for transparent, scalable sentiment classification — powered by XGBoost, GPT, and custom NLP pipelines.

---

## 🔎 Overview

**EESA** integrates traditional machine learning pipelines with modern LLMs like GPT-4 to create explainable, auditable sentiment classifiers. It builds upon the **XSXGBoost** algorithm from the original research and augments it with:

- ✅ Few-shot GPT-based explainability scoring
- ✅ Ensemble boosting via XGBoost and weak classifiers
- ✅ A modular, CLI-controlled architecture
- ✅ Full pipeline serialization and inference
- ✅ Support for real-world datasets like Amazon, Yelp, and IMDB

---

## 📁 Directory Structure

```
eesa/
├── eesa.py                 # CLI entry point
├── pipeline.py             # Core sklearn pipeline logic
├── preprocessing/          # Tokenization, vectorization, LLM injectors
├── ensemble/               # XGBoost, GPT Classifier, Weak Models, ANOVA
├── openai_llm/             # GPT-based scoring and summarization
├── labeled_data/           # LLM-labeled versions of datasets
├── results/                # Training outputs, graphs, analysis CSVs
├── data/                   # Raw datasets (.csv, STS, IMDb, etc.)
├── util.py                 # Logging, color printing, helpers
└── README.md               # You're reading this!
```

---

## 🔧 Installation

```bash
git clone https://github.com/ginkorea/eesa.git
cd eesa
python3 -m venv .eesa_venv
source .eesa_venv/bin/activate
pip install -r requirements.txt
```

### 📌 OpenAI API Key

Set your key via:

```bash
export OPENAI_API_KEY=sk-xxx
```

Or use a `.env` file.

---

## 🚀 Quickstart

### 🧠 Train an Ensemble Model

```bash
python eesa.py train data/gold.csv gold_model --llm --weak --depth 5 --save_dir results
```

- Uses GPT + weak classifiers + XGBoost
- Saves: `results/gold_model_pipeline.pkl` & `gold_model_xgb_model.pkl`

### 🪄 Label a Dataset (with GPT)

```bash
python eesa.py label data/gold.csv 100 0
```

- Labels rows 0–99 with GPT and outputs `labeled_data/gold_labeled.csv`

### 📊 Compare Weak Classifiers (ANOVA)

```bash
python eesa.py compare labeled_data/gold_labeled.csv
```

- Compares NB, SVM, RF, and LR with cross-validation & ANOVA

### 🔮 Predict with Trained Model

```bash
python eesa.py infer results/gold_model_pipeline.pkl data/gold.csv --output_path results/predictions.csv
```

- Adds `prediction` column to your data

---

## 🧠 Methodology

**XSXGBoost + GPT** uses:

- ✅ `XV = [Sentiment Score, Confidence, Explanation Quality]` from GPT
- ✅ Injected as features into XGBoost
- ✅ Explanations stored and graded by GPT

All classification includes a 4-part GPT response:
```
Sentiment Score | Confidence | Explanation Quality | Explanation Text
```

Few-shot examples are included to increase output consistency.

---

## 📊 Datasets Used

- 🛍️ Amazon
- 🎥 IMDB
- 🍽️ Yelp
- 🐦 STS-Gold (Stanford Twitter Sentiment)
- 🎞️ Movie Review v2.0

Located in the `data/` folder.

---

## 📖 Original Research

- 🧾 [Gompert: Explainable Sentiment Analysis (PDF)](https://github.com/ginkorea/eesa/blob/main/research/gompert_paper.pdf)
- 📊 [Slides: Explainable Sentiment Analysis via GPT (PPTX)](https://github.com/ginkorea/eesa/blob/main/research/Gompert_AML_v2.pptx)

Published as part of advanced machine learning work at Johns Hopkins University.

---

## 🛠 Future Work

- [ ] Dash frontend for model exploration
- [ ] LLM benchmarking and output auditing tools
- [ ] Multilingual sentiment support
- [ ] Semi-supervised LLM pre-labeling from web data

---

## 📄 License

MIT License © 2023 Joshua Gompert

---

## 📣 Citation

If you use this repo for your own work:

```
@software{gompert2023eesa,
  author = {Josh Gompert},
  title = {EESA: Ensemble-Based Explainable Sentiment Analysis},
  year = {2023},
  url = {https://github.com/ginkorea/eesa}
}
```
