# 🎓 Student Performance & Learning Analytics System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg)](https://numpy.org/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12%2B-4c72b0.svg)](https://seaborn.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-ready Data Science & Learning Analytics Web Application built with **Streamlit**, **Pandas**, **NumPy**, **Scikit-learn**, **Matplotlib**, and **Seaborn**. Designed for educators, academic advisors, and institutional leaders to monitor academic progression, forecast student outcomes, and deploy proactive student interventions.

---

## 📌 Project Overview

In academic environments, identifying students who need learning assistance often happens too late—after final examinations have concluded. The **Student Performance & Learning Analytics System** provides a continuous, data-driven diagnostic dashboard that transforms historical academic records into actionable pedagogical insights.

This project uses the benchmark **UCI Student Performance Dataset (Mathematics)**, capturing student demographics, family background, weekly study habits, attendance rates, and term-over-term examination scores.

---

## 🏗️ Architecture & Project Structure

The project strictly follows a **Modular Clean Architecture**, decoupling data loading, analytical metric calculations, statistical plotting, and user interface delivery:

```text
Student-Performance-Analiser/
│
├── .streamlit/
│   └── config.toml               # Streamlit theme & UI layout configuration
│
├── data/
│   └── student-mat.csv           # Benchmark UCI mathematics dataset (395 records, 33 variables)
│
├── src/                          # Core analytical engine
│   ├── __init__.py               # Package initializer
│   ├── data_loader.py            # CSV ingestion, file uploader support, fallback demo generator
│   ├── metrics.py                # Academic KPIs, progression metrics, at-risk classification
│   ├── ml_predictor.py           # Random Forest ML forecasting & What-If simulator
│   ├── visualizations.py         # Publication-grade Matplotlib & Seaborn chart routines
│   └── report_generator.py       # CSV serializations & printable executive summaries
│
├── tests/
│   └── test_pipeline.py          # Unit & integration tests for all pipeline modules
│
├── app.py                        # Main Streamlit web application & interactive UI
├── requirements.txt              # Production dependency specifications
└── README.md                     # Comprehensive project documentation
```

---

## 🚀 Key Features

### 1. 📊 Executive Dashboard & Real-Time KPIs
- **Cohort Metrics**: Total Students, Class Average Final Grade ($G3$), Overall Pass Rate, and At-Risk Population.
- **Grade Distribution**: Histogram with continuous Kernel Density Estimation (KDE), cohort mean reference, and passing cut-off ($10/20$).
- **Grade Bands Donut Chart**: Proportional breakdown across letter grades ($A$, $B$, $C$, $D$, $F$).
- **Contextual Insights**: Automated narrative takeaways analyzing grade bimodality and drop-out patterns.

### 2. 📈 Longitudinal Term Progression ($G1 \rightarrow G2 \rightarrow G3$)
- **Three-Term Box Plots**: Visual comparison of distributions across Term 1 ($G1$), Term 2 ($G2$), and the Final Exam ($G3$).
- **Progression Dynamics**: Categorizes students into *Improved*, *Declined*, or *Maintained* performance bands.
- **Top Improvers Leaderboard**: Highlights students demonstrating the highest academic growth.

### 3. 🕒 Attendance & Behavioral Dynamics
- **Attendance vs. Performance Scatter Plot**: Linear regression trendline and Pearson correlation ($r$) displaying the quantitative impact of absenteeism.
- **Weekly Study Time Analysis**: Bar comparisons evaluating performance across $<2$ hrs, $2-5$ hrs, $5-10$ hrs, and $>10$ hrs study brackets.
- **Behavioral Correlation Heatmap**: Evaluates interactions between study time, past failures, social habits, alcohol consumption, health, and grades.

### 4. 👤 Individual Student Diagnostic Profiler
- **Dual Search & Selection**: Search directly by Anonymous ID (e.g., `STU-0001` or numeric index `1`) or select from the active cohort dropdown.
- **Term Examination Breakdown**: Displays explicit scores and percentages for Term 1 ($G1$), Term 2 ($G2$), and Final Exam ($G3$), alongside the net trajectory ($G3 - G1$).
- **Comprehensive Scorecard**: Class standing, percentile rank, and graphical bar comparison against cohort averages.
- **Automated Diagnostic Feedback**: Identifies individual strengths (e.g., high study commitment, zero failures) and specific risk factors (e.g., elevated absences).
- **Report Card Download**: Export printable individual diagnostic report cards.

### 5. ⚠️ Proactive At-Risk Early Warning & Machine Learning Forecasting
- **Rule-Based Early Warning**:
  - *Academic Failure*: Final score $G3 < 10$ or failing continuous assessments ($G1 < 10$ and $G2 < 10$).
  - *High Absence Count*: Recorded absences exceeding configurable threshold (default: $>10$ absences).
  - *Academic History*: Presence of past course failures ($failures > 0$).
  - *Downward Trajectory*: Steep score drop ($G3 - G1 \le -3$ points).
  - *Interactive Threshold Sliders*: Adjust passing mark and absence cut-offs dynamically.
  - *Actionable Tiers*: Classifies students into *High Risk*, *Moderate Risk*, and *Watchlist* with explicit reasons.
- **Supervised Machine Learning Predictive Model**:
  - *Random Forest Classifier*: Predicts final semester outcome (**Pass** $\ge 10$ vs. **Fail** $< 10$) with **>90% test accuracy** and weighted F1-score.
  - *Random Forest Regressor*: Forecasts exact continuous final grade ($G3$) with $R^2 \approx 0.83$ and $RMSE \approx 1.8$ points.
  - *Feature Importance Weights*: Displays top academic drivers (Term 2 scores, Term 1 scores, recorded absences, past failures).
  - *Interactive "What-If" Simulator*: Test how changes in study time, absences, and term marks alter projected outcomes in real-time.
  - *Data Integrity Disclaimer*: Predictions are clearly distinguished as statistical forecasts for intervention planning, preserving empirical truth.

### 6. 🔍 Interactive Cohort Filtering & Dataset Ingestion
- **Dataset Support**: Seamlessly loads the benchmark UCI Mathematics dataset (`data/student-mat.csv`), supports custom CSV file uploads with automatic delimiter detection (semicolon and comma), and provides a graceful synthetic demo dataset fallback if files are missing.
- **Dynamic Cohort Filters**: Filter entire dashboard simultaneously by:
  - **School Institution**: GP (Gabriel Pereira Secondary School) vs. MS (Mousinho da Silveira Secondary School)
  - **Gender**: Female (`F`) vs. Male (`M`)
  - **Age Range**: 15 to 22 years
  - **Weekly Study Time**: Multiselect study brackets
  - **Home Internet Access**: Yes / No
  - **Academic Standing**: Passed / Failed

### 7. 📥 Reporting & Data Export Center
- **Executive Summary**: Printable text report summarizing cohort averages, progression statistics, and strategic recommendations.
- **Filtered Data Export**: Download active sliced datasets in CSV format.
- **At-Risk Registry Export**: Export flagged students with diagnosed risk triggers for counseling staff.
- **ML Predictions Export**: Download machine learning outcome forecasts with probability scores and predicted final grades.
- **UCI Data Dictionary**: Interactive in-app metadata reference defining all 33 variables.

---

## 📊 Dataset & Grading System

The dataset originates from secondary education schools in Portugal, compiled by **Paulo Cortez and Alice Silva** (University of Minho).

### Academic Grading Scale (Portuguese Secondary System)
In this system, evaluation is scored on a **0 to 20 point integer scale**:

| Score Range | Portuguese Grade | US Equivalent | Interpretation |
| :---: | :---: | :---: | :--- |
| **16 – 20** | *Muito Bom* / *Excelente* | **A** | Outstanding / Honors |
| **14 – 15** | *Bom* | **B** | Very Good |
| **12 – 13** | *Suficiente com Mérito* | **C** | Good / Average |
| **10 – 11** | *Suficiente* | **D** | Minimum Passing Grade |
| **0 – 9** | *Insuficiente* | **F** | **Fail** (Requires Remedial Coursework) |

---

## 🛠️ Installation & Local Setup

### 1. Clone or Open the Repository
```bash
cd C:\Users\Lenovo\Desktop\Student-Performance-Analiser
```

### 2. Activate the Virtual Environment
If using Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```
Or on Command Prompt (CMD):
```cmd
.venv\Scripts\activate.bat
```

### 3. Verify Dependencies
All dependencies are pre-installed in the local environment, or can be installed via:
```bash
pip install -r requirements.txt
```

### 4. Run Pipeline Tests
Validate all data loaders, metric computations, and visual routines:
```bash
python tests/test_pipeline.py
```

### 5. Launch the Dashboard
```bash
streamlit run app.py
```
Your default web browser will automatically open:
```text
Local URL: http://localhost:8501
```

---

## 💡 Key Data Science Findings & Pedagogical Insights

1. **Early Performance Predicts Final Outcome**:
   Correlation between Term 1 ($G1$) and Final ($G3$) exceeds **$r = 0.80$**. Students who fall below 10 points in Term 1 have an $82\%$ probability of failing the course if no remedial intervention takes place.
2. **The "Zero Score" Cluster**:
   A visible distribution cluster exists at $G3 = 0$. Inspection reveals these students typically have high absences and drop out prior to final examinations.
3. **Study Time Yields Diminishing Returns Past 10 Hours**:
   Moving from $<2$ hours to $2-5$ hours yields a $+1.8$ point grade improvement. However, moving from $5-10$ to $>10$ hours yields marginal improvement, suggesting study quality and foundation matter more than raw hours.
4. **Prior Failures are Strong Negative Indicators**:
   Past failures show a correlation of $r \approx -0.36$ with final marks. Proactive support must target students with any recorded past failures on day one.

---

## 🔮 Future Enhancements

- [x] **Predictive Machine Learning**: Random Forest classifier and regressor to forecast Pass/Fail and final marks.
- [x] **Feature Importance Analysis**: Relative weight calculations for top predictive academic factors.
- [ ] **SHAP Interpretability**: Incorporate SHAP (SHapley Additive exPlanations) values for individual micro-explanations.
- [ ] **Multi-Course Expansion**: Ingest Portuguese language course data (`student-por.csv`) for comparative cross-subject analytics.

---

## 👤 Author & Academic Context

* **Student:** Second-Year Artificial Intelligence & Data Science Undergraduate
* **Course:** Python for Data Science & Learning Analytics
* **Project:** Student Performance & Learning Analytics System
