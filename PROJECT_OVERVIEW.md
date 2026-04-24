# NovaStar Telecom VOC Survey Analysis

## Live Dashboard

[NovaStar VOC Analytics Dashboard](https://your-app-url.streamlit.app) -- replace with deployed URL

---

## Project Summary

NovaStar Telecom surveys customers after every service interaction, but has no systematic way to prioritize which dissatisfied customers to contact first. This project analyzes 120,000 synthetic post-interaction survey responses, models Detractor risk using a Gradient Boosting classifier (AUC 0.81, top-decile lift 1.82x), and identifies the CX drivers and segments that explain the most variance in NPS outcomes. The result is a prioritized recovery playbook with a quantified revenue opportunity exceeding $190M in Detractor CLTV.

---

## Business Problem

A Customer Insights Data Analyst at a VOC analytics firm answers: "Which customers are most likely to leave, why, and what should we do about it?" This project operationalizes that question for a telecom client with five specific outputs:

1. Which satisfaction drivers (resolution ease, wait time, FCR) predict Detractor status most strongly?
2. Which channels and interaction types produce the highest Detractor concentrations?
3. How much CLTV is at risk, and what is the ROI of a recovery outreach program?
4. Which customers should the recovery team contact first?
5. How does this framework translate to adjacent industries (specifically healthcare)?

---

## Dataset

**Source:** Synthetic (generated with seed=42 for full reproducibility)
**Rows:** 120,000 post-interaction survey responses
**Date range:** April 2023 to April 2025

**Key columns:**
- Six satisfaction driver scores (1-10 scale): resolution ease, agent professionalism, wait time, first contact resolution, digital experience, billing clarity
- Three behavioral signals: prior contacts (90 days), days since outage, escalation count
- One resolution binary: resolved_flag
- Account attributes: segment, service type, channel, region, plan tier, tenure
- Financial: monthly revenue, annual CLTV
- Target variable: detractor_flag (1 = NPS 0-6, 0 = Passive or Promoter)

See `data_dictionary.md` for complete column documentation including leakage flags.

**Target variable distribution:**
- Detractors: 48,760 (40.6%)
- Promoters: 46,371 (38.6%)
- Passives: 24,869 (20.7%)

---

## Methodology

**1. Data generation**
Synthetic data is generated using `numpy.random.default_rng(42)`. Driver scores are drawn from Normal distributions with segment and channel-specific modifiers that create realistic correlations. The Detractor flag is modeled as a logistic function of a weighted composite driver score plus behavioral signals. This produces a dataset with an AUC of 0.81 -- realistic for a survey-based model.

**2. Exploratory analysis**
Driver score gaps between Detractor and Promoter populations were computed for all six drivers. Channel and interaction reason cross-tabs were used to identify high-friction combinations. Repeat contact frequency bands were analyzed for Detractor rate escalation.

**3. Modeling**
Gradient Boosting Classifier trained on 12 features (6 driver scores, 4 behavioral signals, tenure, monthly revenue). Leakage columns excluded: annual_cltv, risk_score, nps_score, nps_category. Train/test split: 80/20 stratified. Validated metrics: AUC 0.81, top-decile lift 1.82x, top-2 decile cumulative gain 36.3%.

**4. Financial impact framework**
Annual CLTV is estimated as 12 months of forward revenue per account adjusted for tenure. Total Detractor CLTV is the recovery opportunity ceiling. The revenue simulator multiplies total Detractor CLTV by a save rate and subtracts contact cost to compute net value.

**5. Dashboard design**
Six-tab Streamlit dashboard with persistent KPI header, sidebar filters, and expandable chart descriptions on every tab. All metrics are computed from the dataset at runtime (no hardcoded financial figures).

---

## Key Findings

1. **Resolution ease is the dominant driver.** Feature importance: 30.3%. Detractors score 5.8/10 on resolution ease vs. 8.2/10 for Promoters -- a 2.4-point gap. This is the single highest-ROI lever for CX improvement programs.

2. **Call Center drives disproportionate Detractor volume.** Call Center interactions produce the highest Detractor rate among all channels. Wait time and FCR scores are systematically lower for Call Center contacts due to volume and queue dynamics.

3. **Repeat contacts are the clearest friction signal.** Customers with 3+ contacts in 90 days have a Detractor rate more than 2x the rate for first-time contacts. A repeat-contact early warning flag in the CRM would give agents immediate context before the interaction begins.

4. **The model concentrates Detractor risk effectively.** Top decile lift of 1.82x means contacting the top 10% of scored customers recovers nearly twice the share of Detractors compared to a random draw. Top-2 decile cumulative gain of 36.3% means 36% of all Detractors can be identified by contacting only 20% of the scored population.

5. **The recovery program is financially viable at multiple save rate assumptions.** At a 20% save rate and $25 per contact, the program generates substantial net value after costs. The revenue simulator allows clients to model their own assumptions.

---

## Technical Stack

- Language: Python 3.12
- Modeling: scikit-learn (GradientBoostingClassifier)
- Dashboard: Streamlit 1.35, Plotly 5.22
- Data: pandas, numpy
- SQL: ANSI SQL (PostgreSQL/BigQuery/Snowflake-compatible)
- Deployment: Streamlit Community Cloud

---

## File Structure

```
novastar-cx-pulse-analysis/
  app.py                    # Primary 6-tab Streamlit dashboard
  01_generate_data.py       # Reproducible dataset generator
  voc_analysis.sql          # 12 queries across 5 sections (ANSI SQL)
  data_dictionary.md        # Column reference with leakage documentation
  requirements.txt          # Pinned Python dependencies
  config.toml               # Streamlit theme configuration
  data/
    voc_survey.csv          # 120,000-row synthetic dataset (seed=42)
  docs/
    PROJECT_OVERVIEW.md     # This file
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Aztexan512/novastar-cx-pulse-analysis
cd novastar-cx-pulse-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset (or use the committed CSV)
python 01_generate_data.py

# 4. Run the dashboard
streamlit run app.py
```

---

## How to Deploy

1. Push the repository to GitHub (ensure data/voc_survey.csv is committed)
2. Log in to share.streamlit.io
3. Click "New app" and connect to the GitHub repo
4. Set main file path to `app.py`
5. Click "Deploy"
6. Add the deployed URL to the README and GitHub About section

---

*Project version 1.0 -- Luciano Casillas | luciano.casillasjr@outlook.com | github.com/Aztexan512*
