# Smart-Recruitment-Assistant
AI-powered candidate screening and talent ranking web app using Machine Learning &amp; Streamlit.
readme_content = """# 💼 Smart Recruitment Assistant

An end-to-end Machine Learning web application designed to automate candidate screening, evaluate career mobility trends, and rank high-potential technical talent to optimize recruitment workflows.

> **Graduation Project** — Information Technology Institute (**ITI**)  
> **Author:** Abdallah Mostafa ([@Abdallah-Mostafa19](https://github.com/Abdallah-Mostafa19))

---

## 📌 Project Overview

Recruitment teams face major bottlenecks when manually sifting through thousands of technical job applications. This leads to extended hiring cycles, screening fatigue, and subjective human bias.

**Smart Recruitment Assistant** solves this by leveraging supervised machine learning to:
1. Predict candidate mobility and willingness to advance to technical interview stages.
2. Generate a normalized **Suitability Score (0% – 100%)**.
3. Automate initial screening decisions (`Advance`, `Hold / Review`, `Archive`).
4. Dynamically rank the top candidates across large applicant pools.

---

## 🚀 Key Features

* **🎯 Single Candidate Screener:** Interactive sliders and dropdown inputs providing real-time probability inference and automated HR decision badges.
* **🏆 Top 10 Talent Surfacing:** Automated ranking pipeline sorting applicants by calibrated model confidence to prioritize outreach.
* **📁 Batch CSV Processor:** Upload complete applicant CSV files for bulk scoring, recommendation assignment, and instant report download ready for ATS integration.

---

## 📊 Dataset & Preprocessing Pipeline

* **Data Source:** HR Analytics: Job Change of Data Scientists dataset (19,158 records).
* **Missing Value Imputation:** Categorical columns (`gender`, `company_size`, `company_type`) imputed with `'Unknown'` to preserve distributional patterns; numerical columns (`experience`) imputed using median values.
* **Feature Engineering:**
  * Cleaned and converted string ranges (`>20`, `<1`) into continuous numeric values.
  * Encoded job tenure intervals (`>4`, `never`) into integer scales.
  * Mapped probability scores into calibrated HR operational thresholds.
* **Encoding & Scaling:** One-Hot Encoding (`pd.get_dummies` with `drop_first=True` to prevent dummy variable traps) and `StandardScaler` for numeric feature normalization.
* **Class Imbalance Handling:** Addressed the 75/25 target class skew using stratified splitting and class-weighted cost functions.

---

## 🤖 Model Performance & Benchmark

Two primary algorithms were evaluated on a stratified 80/20 test partition:

| Evaluation Metric | Logistic Regression (Balanced) | Random Forest Classifier | Production Decision |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 76.8% | **79.4%** | **Random Forest (+2.6%)** |
| **Precision (Class 1)** | 0.52 | **0.59** | **Random Forest (+0.07)** |
| **Recall (Class 1)** | **0.65** | 0.58 | Logistic Regression (+0.07) |
| **F1-Score (Class 1)** | 0.58 | 0.58 | Balanced |
| **ROC-AUC Score** | 0.781 | **0.804** | **Random Forest (+0.023)** |

> **Production Choice:** **Random Forest** was selected for production deployment in Streamlit due to its superior discriminative capacity (0.804 ROC-AUC) and significantly lower false positive rate, ensuring high precision during live applicant ranking.

---

## 📈 Key Business Intelligence (BI) Insights

1. **City Development Index (CDI):** Accounts for over **25.5%** of the decision weight. Candidates from emerging urban hubs (< 0.70) are 3.2x more receptive to job transitions compared to developed metro areas.
2. **Training Hours Completed:** Accounts for **21.2%**; higher commitment to continuous learning strongly indicates readiness for career progression.
3. **Experience & Stability:** Mid-level professionals (4–8 years of experience) with established tenure represent the optimal fit between skill maturity and retention likelihood.

---

## 📂 Repository Structure

```text
├── app.py                                         # Interactive Streamlit application
├── SmartRecruitment.ipynb                         # Model training, analysis & benchmark notebook
├── recruitment_model.pkl                          # Serialized Random Forest model
├── model_features.pkl                             # Feature names list for schema alignment
├── aug_train.csv                                  # Training dataset
├── requirements.txt                               # Project dependencies
├── Smart_Recruitment_Performance_Documentation.pdf # Comprehensive technical report
├── Smart_Recruitment_Verified_Final.pptx          # Final presentation slide deck
└── README.md                                      # Project documentation
