# NovaStar Telecom VOC Survey Analysis

 ---
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GradientBoosting-F7931E?logo=scikit-learn&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-PostgreSQL%20%2F%20Snowflake-4169E1?logo=postgresql&logoColor=white)
![NPS](https://img.shields.io/badge/NPS-Voice%20of%20Customer-0077B3)
![CLTV](https://img.shields.io/badge/CLTV-Financial%20Modeling-08CAA9)
![HCAHPS](https://img.shields.io/badge/HCAHPS-Healthcare%20Application-4169E1)
![Dataset](https://img.shields.io/badge/Dataset-120K%20Records-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

---

**40.6% of post-interaction customers become Detractors. This project identifies which ones to contact first, why they scored low, and how to build the business case for a recovery program -- using 120,000 survey responses, a Gradient Boosting model (AUC 0.81), and a live Streamlit dashboard.**

---

## Live Dashboard

[NovaStar VOC Analytics Dashboard](https://l4jk7jnpschi6ot7do7zgq.streamlit.app/)

> For a full walkthrough of the methodology, findings, and business recommendations, see the **[Project Overview](PROJECT_OVERVIEW.md)**.

---

## Project Summary

This portfolio project demonstrates the full VOC analytics workflow a Customer Insights Data Analyst would deliver for a consulting client:

- Synthetic dataset of 120,000 post-interaction survey responses (seed=42, fully reproducible)
- Driver importance analysis identifying resolution ease (30.3%), wait time (19.7%), and first contact resolution (19.1%) as the primary NPS predictors
- Gradient Boosting Detractor risk model: AUC 0.81, top-decile lift 1.82x, top-2 decile cumulative gain 36.3%
- Revenue simulator quantifying CLTV at risk and net value of recovery outreach at different save rate assumptions
- Cross-industry translation mapping the telecom VOC framework to healthcare patient experience (HCAHPS) analytics

---

## Key Findings

1. Resolution ease is the dominant driver of NPS Detractor status (30.3% model importance). Detractors score 5.8/10 vs. 8.2/10 for Promoters -- a 2.4-point gap that represents the single highest-ROI CX improvement lever.
2. Customers who contacted NovaStar 3+ times in 90 days show a Detractor rate more than 2x the baseline. Repeat contact is the most actionable pre-survey friction signal.
3. Call Center interactions drive the highest Detractor concentration across all channels, primarily through lower wait time and FCR scores.
4. The top 20% of scored customers contain 36.3% of all Detractors -- the model concentrates risk effectively for targeted outreach programs.

---

## Technical Stack

- Python 3.12, scikit-learn, pandas, numpy
- Streamlit 1.35, Plotly 5.22
- SQL: ANSI-compatible (PostgreSQL/BigQuery/Snowflake)
- Deployment: Streamlit Community Cloud

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/Aztexan512/novastar-cx-pulse-analysis
cd novastar-cx-pulse-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset (or use the committed CSV directly)
python 01_generate_data.py

# 4. Run the dashboard
streamlit run app.py
```

---

## File Structure

```
novastar-cx-pulse-analysis/
  app.py                    # 6-tab Streamlit dashboard
  01_generate_data.py       # Reproducible dataset generator
  voc_analysis.sql          # 12 queries across 5 sections (ANSI SQL)
  data_dictionary.md        # Column reference with leakage documentation
  requirements.txt          # Pinned Python dependencies
  config.toml               # Streamlit theme configuration
  data/
    voc_survey.csv          # 120,000-row synthetic dataset (seed=42)
  PROJECT_OVERVIEW.md        # Full methodology and findings
```

---

## Author

**Luciano Casillas** | Independent Analytics Consultant
luciano.casillasjr@outlook.com | [GitHub](https://github.com/Aztexan512) | [LinkedIn](https://linkedin.com/in/luciano-casillas)
