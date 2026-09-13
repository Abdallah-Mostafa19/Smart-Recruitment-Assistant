# Smart-Recruitment-Assistant

An end-to-end Machine Learning web application designed to automate candidate screening, evaluate career mobility trends, and rank high-potential technical talent to optimize the recruitment funnel.

---

## 📌 Project Overview
Recruitment teams frequently face massive application backlogs, high screening overhead, and subjective review fatigue. **Smart Recruitment Assistant** addresses this by using machine learning models to score candidate suitability, predict transition likelihood to technical interview stages, and surface the top-performing candidates automatically.

---

## 🚀 Key Features
* **Individual Candidate Screener:** Interactive sliders and dropdowns providing real-time suitability percentages and automated decision badges (`Advance`, `Hold / Review`, `Archive`).
* **Top 10 Talent Surfacing:** Automated ranking pipeline extracting the highest-fit profiles from the applicant pool.
* **Batch CSV Processing:** Upload entire applicant datasets for bulk scoring, recommendation labeling, and exportable CSV reports ready for ATS integration.

---

## 📊 Dataset & Preprocessing Pipeline
* **Source:** HR Analytics: Job Change of Data Scientists dataset (19,158 records).
* **Imputation:** Categorical missing entries preserved via `'Unknown'` tokens; numeric gaps imputed using the median.
* **Feature Engineering:** String ranges for experience (`>20`, `<1`) and last job gaps (`>4`, `never`) converted into continuous numeric metrics.
* **Encoding & Scaling:** One-Hot Encoding (`pd.get_dummies` with `drop_first=True`) and `StandardScaler` for numeric feature normalization.
* **Class Imbalance:** Handled using stratified data partitioning and class-weight balancing.

---

## 🤖 Model Performance & Evaluation

Two primary classification algorithms were trained and benchmarked on a stratified 80/20 test split:

| Evaluation Metric | Logistic Regression (Balanced) | Random Forest Classifier | Production Choice |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 76.8% | **79.4%** | **Random Forest (+2.6%)** |
| **Precision (Class 1)** | 0.52 | **0.59** | **Random Forest (+0.07)** |
| **Recall (Class 1)** | **0.65** | 0.58 | Logistic Regression (+0.07) |
| **F1-Score (Class 1)** | 0.58 | 0.58 | Balanced |
| **ROC-AUC Score** | 0.781 | **0.804** | **Random Forest (+0.023)** |

> **Verdict:** Random Forest was selected for the live Streamlit engine due to its superior discriminative power (0.804 ROC-AUC) and lower false alarm rate in candidate ranking.

---

## 📈 Business Intelligence Insights
1. **City Development Index (CDI):** Emerged as the strongest predictor of candidate transition (~25.5% relative importance). Applicants from developing urban centers (< 0.70) are over 3x more receptive to career transitions.
2. **Training Hours:** High training hours (≥ 40 hrs) strongly correlate with candidate commitment and qualification alignment.
3. **Experience & Job Tenure:** Mid-level experience candidates (4–8 years) with ≥ 1 year at their prior employer offer the optimal balance of immediate productivity and career growth readiness.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Web Deployment:** Streamlit
* **Model Serialization:** Pickle / Joblib

---

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash git clone https://github.com/Abdallah-Mostafa19/Smart-Recruitment-Assistant.git
cd Smart-Recruitment-Assistant

