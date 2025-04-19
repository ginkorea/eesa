


# 🧠 EESA: Ensemble-Based Explainable Sentiment Analysis

> A hybrid NLP/ML/LLM framework for transparent, scalable sentiment classification — powered by XGBoost, GPT, and custom NLP pipelines.

---

## 🔎 Overview

**EESA** integrates traditional machine learning pipelines with modern LLMs like GPT-4 to create explainable, auditable sentiment classifiers. It builds upon the **XSXGBoost** algorithm and augments it with:

- ✅ Few-shot GPT-based explainability scoring  
- ✅ Ensemble boosting via XGBoost and weak classifiers  
- ✅ A modular, CLI-controlled architecture  
- ✅ Full pipeline serialization and inference  
- ✅ Support for real-world datasets like Amazon, Yelp, and IMDB  

---

## 📁 Project Structure

```
eesa/
├── eesa.py                 # CLI entry point
├── pipeline.py             # Core sklearn pipeline logic
├── preprocessing/          # Tokenization, vectorization, LLM injectors
├── ensemble/               # XGBoost, weak classifiers, ANOVA analysis
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

---

## 📌 OpenAI API Key Setup

You can provide your OpenAI API key in one of two ways:

### 1. 🔒 Preferred (Environment Variable)

```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

This is the most secure method and is recommended for deployment or version-controlled environments.

### 2. 🗂️ Alternative (Local Python File)

1. Open the file:  
   `openai_llm/_key.py`

2. Replace the placeholder:

   ```python
   my_key = "sk-proj-replace-this-with-your-actual-api-key"
   ```

3. Rename the file:  
   `openai_llm/key.py`

> ⚠️ Never commit your real key to GitHub.  
> 🔑 Treat your API key like a password — keep it private.

---

## 🚀 Quickstart

### 🧠 Train an Ensemble Model

```bash
python eesa.py train data/gold.csv gold_model --llm --weak --depth 5 --save_dir results
```

- Uses GPT + weak classifiers + XGBoost
- Saves results to:  
  `results/gold_model_pipeline.pkl` and `gold_model_xgb_model.pkl`

---

### 🪄 Label a Dataset (GPT-based)

```bash
python eesa.py label data/gold.csv 100 0
```

- Labels rows 0–99 using GPT
- Saves output to: `labeled_data/batch_0_100.csv`

---

### 📊 Compare Weak Classifiers (ANOVA)

```bash
python eesa.py compare labeled_data/gold_labeled.csv
```

- Compares NB, SVM, RF, and LR using k-fold CV + ANOVA

---

### 🔮 Inference with Trained Pipeline

```bash
python eesa.py infer results/gold_model_pipeline.pkl data/gold.csv --output_path results/predictions.csv
```

- Appends a `prediction` column to the input CSV

---

## 🧪 Methodology

### GPT + XSXGBoost

EESA uses a custom ensemble pipeline that combines:

- `XV = [Sentiment Score, Confidence, Explanation Quality]` from GPT  
- Injected directly as features into XGBoost  
- GPT also generates natural language rationales ("explanations")

GPT returns the following format per input:

```
Sentiment Score | Confidence | Explanation Quality | Explanation
```

These outputs are:
- Averaged across few-shot variants
- Scored for explainability
- Stored alongside predictions for full transparency

---

## 📊 Datasets Supported

- 🛍️ Amazon Reviews  
- 🎥 IMDB  
- 🍽️ Yelp  
- 🐦 STS-Gold (Stanford Twitter Sentiment)  
- 🎞️ Movie Review v2.0  

→ Place raw datasets in the `data/` directory.

---

## 📖 Original Research

- 📄 [Paper (PDF)](https://github.com/ginkorea/eesa/blob/main/research/gompert_paper.pdf)  
- 📽️ [Presentation Slides (PPTX)](https://github.com/ginkorea/eesa/blob/main/research/Gompert_AML_v2.pptx)

Published as part of a master’s thesis for the Advanced Machine Learning program at Johns Hopkins University.

---

## 🧭 Roadmap

- [ ] Interactive Dash frontend for exploring predictions  
- [ ] LLM benchmarking + hallucination audit tools  
- [ ] Multilingual sentiment support  
- [ ] Semi-supervised auto-labeling from live web content  
- [ ] Hugging Face-compatible export  

---

## 📄 License

MIT License © 2023–2025 Joshua Gompert

---

## 📣 Citation

```
@software{gompert2023eesa,
  author = {Josh Gompert},
  title = {EESA: Ensemble-Based Explainable Sentiment Analysis},
  year = {2023},
  url = {https://github.com/ginkorea/eesa}
}   
```