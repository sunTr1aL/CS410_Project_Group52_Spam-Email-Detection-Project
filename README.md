# 📧 Spam Email Detection Project
This project is the github repository for CS410 Final Project (Group 52) at UIUC.

A comprehensive spam classification pipeline combining **traditional NLP models** (Naive Bayes, Logistic Regression, SVM) with **modern deep learning** (DistilBERT and RoBERTa).  
The project performs full-cycle data processing: loading → cleaning → vectorization → modeling → evaluation, complete with EDA reports and confusion matrices.

---

## 🧠 Overview

This repository provides a comprehensive framework for **spam email detection** and experimentation. It includes:

- 📊 **Data Standardization** — unified preprocessing across major public datasets:
  - Enron Email Dataset  
  - SpamAssassin  
  - TREC 2007  

- ⚙️ **Baseline Models (Linear SVM, Logistic Regression, Narive Bayes)** — classic machine learning benchmarks for comparison and reproducibility.

- 🤖 **Transformer-Based Classifier (DistilBERT, RoBERTa)** — fine-tuned for advanced semantic understanding of email content.

- 📈 **Automated Reporting** — generates detailed evaluation metrics in CSV format and visual **confusion matrices**.

For additional transformer experiments and RoBERTa-based results, please visit:  
🔗 [RoBERTa Code & Results Repository](https://github.com/yueqiangwu/CS409_final_project)

---

## 🧩 Tested Datasets

| Dataset        | Description |
|----------------|--------------|
| **Enron**      | Corporate email dataset with ham/spam labels |
| **SpamAssassin** | Classic spam corpus with clear spam features |
| **TREC 2007**  | Research dataset including sender metadata |

To use your own dataset, just place a CSV under `data/raw/` and modify the loader in `baseline_eval.py`.

📄 **Metrics Source:**  
All test results are included in:
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




