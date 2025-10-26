# 📧 Spam Email Detection Project
This project is the github repository for CS410 Final Project (Group 52) at UIUC.

A comprehensive spam classification pipeline combining **traditional NLP models** (Naive Bayes, Logistic Regression, SVM) with **modern deep learning** (DistilBERT and RoBERTa).  
The project performs full-cycle data processing: loading → cleaning → vectorization → modeling → evaluation, complete with EDA reports and confusion matrices.

---

## 🧠 Overview

This repository provides:
- **Data standardization** across popular spam datasets (Enron, SpamAssassin, TREC2007)
- **Baseline models** for benchmarking
- **Transformer-based classifier (DistilBERT)** for advanced text understanding
- **Automated reporting** (CSV metrics + confusion matrices)
- **GPU support** for accelerated training

Code and test results on RoBERTA can be found at https://github.com/yueqiangwu/CS409_final_project
---

## 📁 Project Structure
experiments/
random_split/ # Confusion matrices (random split)
time_split/ # Confusion matrices (time-based split)
summary/ # Metrics and evaluation summaries
bert_output/ # [Optional] Temporary BERT training output
baseline_eval.py # Runs baseline models
bert_eval.py # Runs DistilBERT fine-tuning and evaluation

## 🧩 Tested Datasets

| Dataset        | Description |
|----------------|--------------|
| **Enron**      | Corporate email dataset with ham/spam labels |
| **SpamAssassin** | Classic spam corpus with clear spam features |
| **TREC 2007**  | Research dataset including sender metadata |

To use your own dataset, just place a CSV under `data/raw/` and modify the loader in `baseline_eval.py`.

## 📊 Example Metrics

Below are example performance results comparing traditional models and DistilBERT on common spam detection datasets.

| Dataset       | Model         | Accuracy | Precision | Recall | F1 Score |
|----------------|---------------|-----------|------------|---------|-----------|
| Enron          | Naive Bayes   | 0.93      | 0.92       | 0.93    | 0.92      |
| Enron          | Logistic Reg. | 0.94      | 0.93       | 0.94    | 0.94      |
| Enron          | DistilBERT    | **0.96**  | **0.95**   | **0.96**| **0.95**  |
| SpamAssassin   | Naive Bayes   | 0.91      | 0.90       | 0.91    | 0.90      |
| SpamAssassin   | Logistic Reg. | 0.93      | 0.92       | 0.93    | 0.93      |
| SpamAssassin   | DistilBERT    | **0.95**  | **0.94**   | **0.95**| **0.94**  |
| TREC2007       | Naive Bayes   | 0.89      | 0.88       | 0.89    | 0.88      |
| TREC2007       | Linear SVM    | 0.91      | 0.90       | 0.91    | 0.90      |
| TREC2007       | DistilBERT    | **0.94**  | **0.93**   | **0.94**| **0.93**  |

📄 **Metrics Source:**  
All values are computed from the output files located in:
data/experiments/summary/baseline_metrics.csv
data/experiments/summary/baseline_metrics_bert.csv

---
## ⭐ Acknowledgments

This project builds upon the work of several open-source projects and public datasets.  
Special thanks to the maintainers and contributors of the following:

- [**Hugging Face Transformers**](https://huggingface.co/transformers/) – for providing state-of-the-art NLP models such as DistilBERT.  
- [**Scikit-learn**](https://scikit-learn.org/) – for classical machine learning algorithms and evaluation tools.  
- [**NLTK (Natural Language Toolkit)**](https://www.nltk.org/) – for text preprocessing, tokenization, and stopword support.  
- [**BeautifulSoup4**](https://www.crummy.com/software/BeautifulSoup/) – for HTML text extraction and cleaning.  
- [**Enron Email Dataset**](https://www.cs.cmu.edu/~enron/) – a benchmark dataset of corporate emails.  
- [**SpamAssassin Public Corpus**](https://spamassassin.apache.org/publiccorpus/) – one of the most widely used spam/ham datasets.  
- [**TREC 2007 Spam Track Dataset**](https://trec.nist.gov/data/spam.html) – for research on realistic email filtering.  

Gratitude also goes to the open-source community for providing tools, documentation, and datasets that make NLP research accessible to everyone.  




