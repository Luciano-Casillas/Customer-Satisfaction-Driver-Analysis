# 📡 NovaStar Telecom VOC Survey Analysis

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GradientBoosting-F7931E?logo=scikit-learn&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-PostgreSQL%2F%20Snowflake-4169E1?logo=postgresql&logoColor=white)
![NPS](https://img.shields.io/badge/NPS-Voice%20of%20Customer-0077B3)
![CLTV](https://img.shields.io/badge/CLTV-Financial%20Modeling-08CAA9)
![HCAHPS](https://img.shields.io/badge/HCAHPS-Healthcare%20Application-4169E1)
![Dataset](https://img.shields.io/badge/Dataset-120K%20Records-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

**40.6% of post-interaction customers become Detractors.** This project analyzes 120,000 survey responses to identify which customers to contact first, why they scored low, and how to build the business case for a recovery program using a Gradient Boosting model (AUC 0.81) and a live Streamlit dashboard.

---

## 📋 Table of Contents

- [Project Background](#-project-background)
- [Executive Summary](#-executive-summary)
- [Insights Deep Dive](#-insights-deep-dive)
- [Recommendations](#-recommendations)
- [Data Structure](#-data-structure)
- [Setup](#-setup)
- [Live Dashboards](#-live-dashboards)
- [File Structure](#-file-structure)
- [Assumptions and Caveats](#-assumptions-and-caveats)
- [Author](#-author)

---

## 🏢 Project Background

NovaStar Telecom surveys customers after every service interaction, collecting satisfaction scores across six experience dimensions alongside behavioral signals and account data. Despite a consistent survey cadence, the company had no systematic way to prioritize which dissatisfied customers to contact first, leaving a large share of at-risk revenue unaddressed between survey receipt and any form of outreach.

The core business problem a Customer Insights Data Analyst answers for a VOC analytics client is: "Which customers are most likely to leave, why, and what should we do about it?" This project turns that question into a structured, evidence-based analysis, delivering five specific findings tied to driver importance, channel performance, repeat contact patterns, model-driven prioritization, and financial impact.

The project delivers a prioritized recovery playbook supported by a Gradient Boosting Detractor risk model, a financial impact simulator, and a cross-industry translation to healthcare patient experience (HCAHPS) analytics. All data is synthetic, generated with seed 42 for full reproducibility, and covers April 2023 through April 2025.

---

## 📊 Executive Summary

- Resolution ease is the single strongest predictor of Detractor status, accounting for **30.3% of model importance**. Detractors score **5.8/10** vs. **8.2/10** for Promoters, a 2.4-point gap representing the highest-ROI CX improvement lever.
- Call Center interactions produce the highest Detractor concentration of any channel. Wait time and first contact resolution (FCR) scores are systematically lower for Call Center contacts, driven by volume and queue dynamics.
- Customers who contacted NovaStar **3 or more times in 90 days** show a Detractor rate more than **2x the overall baseline of 40.6%**, making repeat contact the most actionable pre-survey friction signal.
- The Gradient Boosting model achieves a **top-decile lift of 1.82x** and a top-2 decile cumulative gain of **36.3%**, meaning 36% of all Detractors can be identified by contacting only 20% of the scored population.
- Detractor accounts represent more than **$190M in annual customer lifetime value**. At a 20% save rate and $25 per contact, a targeted recovery program returns approximately **$36.8M in net value** after all outreach costs.
- Resolution ease, wait time, and first contact resolution together explain **69% of Detractor risk**, concentrating the improvement roadmap on three addressable process levers.

---

## 🔍 Insights Deep Dive

### 1. Resolution Ease is the Dominant NPS Driver

Resolution ease accounts for **30.3% of total model feature importance**, more than any other driver. Detractors rate this dimension **5.8/10** compared to **8.2/10** for Promoters, a 2.4-point gap that is both the largest score differential and the highest-leverage improvement target in the entire model. Closing this gap in Billing Dispute and Technical Support interactions would produce the most measurable NPS lift of any single initiative.

<!-- SCREENSHOT REQUIRED: CX Drivers tab -- driver importance bar chart and Detractor vs. Promoter score comparison chart -->
<!-- Add screenshots/cx_drivers.png once available. Suggested capture: Full CX Drivers tab at 1280px viewport, with both the driver importance ranking and the score gap chart visible. -->

### 2. Call Center Channel Drives Disproportionate Detractor Volume

Call Center interactions produce the highest Detractor rate of any channel, driven by wait time and first contact resolution scores that are systematically lower than all other channels. Billing Dispute, Technical Support, Outage Follow-Up, and Cancellation Request interactions each exceed the **40.6% overall Detractor baseline**, with Cancellation Request and Outage Follow-Up among the most concentrated risk combinations. Routing high-risk contacts to senior agents before the interaction begins shifts the program from reactive to preventive.

<!-- SCREENSHOT REQUIRED: CX Drivers tab -- channel Detractor rate chart and interaction reason breakdown -->
<!-- Add screenshots/cx_drivers.png or a separate screenshots/channel_analysis.png once available. Suggested capture: Channel performance chart and interaction reason heatmap on the CX Drivers tab. -->

### 3. Repeat Contact Frequency is the Clearest Pre-Survey Friction Signal

Customers who contacted NovaStar **3 or more times in the prior 90 days** have a Detractor rate more than **2x the overall baseline**. This friction pattern is visible in the CRM before the interaction begins, yet agents currently have no way to surface it. A repeat-contact flag costs nothing to build once the data is connected and gives agents immediate context, allowing them to open with acknowledgment rather than discovery.

<!-- SCREENSHOT REQUIRED: CX Drivers tab -- repeat contact frequency vs. Detractor rate chart -->
<!-- Add screenshots/cx_drivers.png once available. Suggested capture: Repeat contact frequency escalation chart showing Detractor rate by prior_contacts_90d band. -->

### 4. The Model Concentrates Detractor Risk Effectively for Targeted Outreach

The Gradient Boosting model achieves a top-decile lift of **1.82x**, meaning contacting the top 10% of risk-scored customers recovers nearly twice the share of Detractors compared to a random draw. The top-2 decile cumulative gain of **36.3%** confirms the model is an effective prioritization tool: a recovery team contacting only 20% of the scored population captures more than one-third of all Detractors. Priority 1 accounts combine a high risk score with annual CLTV above $800.

<!-- SCREENSHOT REQUIRED: Model + Risk tab -- cumulative gain curve and feature importance chart -->
<!-- Add screenshots/model_risk.png once available. Suggested capture: Model + Risk tab showing cumulative gain/lift chart and feature importance bar chart both visible at 1280px viewport. -->

### 5. The Recovery Program is Financially Viable Under Conservative Assumptions

Detractor accounts represent more than **$190M in annual customer lifetime value**. At a 20% save rate and $25 per contact, total outreach costs approximately $1.2M against $38M in protected revenue, producing approximately **$36.8M in net value**. The financial simulator on the dashboard allows stakeholders to model their own save rate and cost-per-contact assumptions, making the business case adaptable to different program designs.

<!-- SCREENSHOT REQUIRED: Financial Impact tab -- revenue at risk summary and recovery simulator with scenario chart -->
<!-- Add screenshots/financial_impact.png once available. Suggested capture: Financial Impact tab with the revenue at-risk cards and the simulator results card and scenario bar chart all visible. -->

---

## 💡 Recommendations

### Immediate Actions (0-30 Days)

**Deploy risk scoring to the Call Center queue.**
Call Center produces the highest Detractor rate of any channel; routing top-decile customers (1.82x more likely to be Detractors) to senior agents before they score low shifts the program from reactive to preventive with no new technology required.

**Launch a closed-loop recovery program for top-risk Detractors.**
Contacting the top 20% of risk-scored customers recovers 36.3% of all Detractors; at a 20% save rate and $25 per contact, the program returns approximately **$36.8M in net value** against roughly $1.2M in outreach costs.

### Short-Term Actions (30-90 Days)

**Redesign resolution scripts for Billing Dispute interactions.**
Resolution ease accounts for **30.3% of model importance** and shows the largest Detractor/Promoter score gap (2.4 points); empowering agents to resolve billing issues on first contact without supervisor approval for credits up to $50 directly closes this gap.

**Add a repeat-contact flag to the CRM.**
Customers with **3+ contacts in 90 days** show a Detractor rate more than 2x the baseline; a CRM flag surfaced before the interaction begins gives agents immediate context at no incremental build cost once the data is connected.

### Strategic Investments (90+ Days)

**Build a continuous driver importance retraining pipeline.**
The current model reflects a fixed two-year snapshot; Self-Serve App is the second-fastest growing channel in the dataset, and a monthly retraining pipeline ensures CX investment priorities stay aligned with where friction is actually occurring rather than where it was two years ago.

**Pilot the VOC framework in an adjacent industry.**
Every NovaStar satisfaction driver maps directly to a published HCAHPS composite measure; the same risk-scoring, driver importance, and recovery simulation approach applies to readmission prevention and discharge planning quality with minimal adaptation, opening a new service line.

---

## 🗂️ Data Structure

All data in this project is synthetic. The analysis-ready dataset (`data/voc_survey.csv`) was generated from source tables that reflect how this data actually lives in a real telecom system.

**Dataset: 120,000 rows | Seed: 42 | Time window: April 2023 to April 2025**

| Column | Type | Description |
|---|---|---|
| customer_id | string | Unique respondent ID (format: NST-0000001). Not used in modeling. |
| survey_date | date | Date survey was completed (YYYY-MM-DD). Used for trend analysis. |
| customer_segment | string | Customer type: Consumer, Small Business, Enterprise. |
| service_type | string | Product subscribed: Mobile, Broadband, Bundle, TV Only. |
| interaction_channel | string | Channel where interaction occurred: Call Center, In-Store, Chat/Digital, Self-Serve App, Field Tech. |
| interaction_reason | string | Primary reason for the interaction: Billing Dispute, Technical Support, Service Upgrade, Cancellation Request, New Activation, General Inquiry, Outage Follow-Up. |
| region | string | Geographic region: West, Southwest, Midwest, Southeast, Northeast. |
| plan_tier | string | Pricing tier: Basic, Standard, Premium, Business Plus. |
| tenure_months | integer | Months since customer activation (1-144). Used in model training. |
| monthly_revenue | float | Estimated monthly revenue from this account (USD). Dashboard only. |
| annual_cltv | float | Estimated 12-month forward customer lifetime value (USD). Dashboard only. |
| resolution_ease | float | How easy was it to resolve your issue? (1-10 scale). Top-ranked driver: 30.3% model importance. |
| agent_professionalism | float | How professional was the agent or representative? (1-10 scale). |
| wait_time_score | float | Satisfaction with how long you waited (1-10 scale). Second most important driver: 19.7%. |
| first_contact_resolution | float | Was your issue resolved without needing to follow up? (1-10 scale). Third most important: 19.1%. |
| digital_experience | float | Satisfaction with the digital or app experience (1-10 scale). |
| billing_clarity | float | Clarity of your bill and charges (1-10 scale). |
| prior_contacts_90d | integer | Number of contacts with NovaStar in prior 90 days (0-12). |
| days_since_outage | integer | Days since a network outage affected the customer's area (0-90; 90 = no recent outage). |
| resolved_flag | integer (0/1) | 1 = issue was resolved in this interaction; 0 = unresolved. |
| escalation_count | integer | Number of escalations during this interaction (0-3). |
| verbatim_provided | integer (0/1) | 1 = customer submitted a written comment; 0 = did not. |
| nps_score | integer | Raw Net Promoter Score rating (0-10). Not used directly in model. |
| nps_category | string | Detractor, Passive, or Promoter. Derived from nps_score. |
| detractor_flag | integer (0/1) | Binary target variable. 1 = Detractor (NPS 0-6); 0 = Passive or Promoter. Dataset rate: 40.6%. |
| risk_score | float | Composite at-risk score (0-10). Dashboard only. |

**Leakage-prone columns (excluded from model training):**

| Column | Risk | Reason |
|---|---|---|
| annual_cltv | HIGH | Derived from revenue and tenure; incorporates future value information. |
| risk_score | HIGH | Composite score incorporating nps_score; direct leakage of target signal. |
| nps_score | HIGH | Raw source of the target variable detractor_flag. |
| nps_category | HIGH | Derived directly from nps_score; perfect leakage of target. |
| monthly_revenue | MEDIUM | Proxy for plan_tier, which is already included as a training feature. |

Source table schema: See [data/schema/erd.md](data/schema/erd.md) for the entity-relationship diagram and [data/schema/table_definitions.md](data/schema/table_definitions.md) for source table grain and join logic.

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/Aztexan512/Customer-Satisfaction-Driver-Analysis.git
cd Customer-Satisfaction-Driver-Analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the primary dashboard
streamlit run app.py
```

> Note: The analysis-ready dataset is committed to this repo at `data/voc_survey.csv`. No data generation step is required to run the dashboard or notebook.

---

## 🚀 Live Dashboards

| Dashboard | Link |
|---|---|
| NovaStar VOC Analytics Explorer | [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cttgrkprwzdmmrdgg82cvs.streamlit.app/) |

<!-- STREAMLIT URL: Primary dashboard URL is active above. If a benchmarking dashboard is deployed, add a second row with its URL. -->

---

## 📁 File Structure

```
Customer-Satisfaction-Driver-Analysis/
|-- README.md                         # This file
|-- app.py                            # Primary Streamlit dashboard (6 tabs)
|-- voc_analysis.sql                  # 12 queries across 5 business question sections
|-- requirements.txt                  # Pinned Python dependencies
|-- runtime.txt                       # Python version pin for Streamlit Cloud
|-- config.toml                       # Streamlit theme configuration
|-- .gitignore                        # Excludes build artifacts, secrets, local files
|-- data/
|   |-- voc_survey.csv                # Analysis-ready dataset (120,000 rows, seed=42)
|-- data_dictionary.md                # Column reference with leakage documentation
|-- PROJECT_OVERVIEW.md               # Methodology, key findings, how to run
```

---

## ⚠️ Assumptions and Caveats

**Synthetic data:** All data in this project is synthetic. The dataset was generated using NumPy with seed 42 for reproducibility. It is designed to produce realistic analytical patterns but does not represent any real company, customer, or transaction.

**Modeling assumptions:**
- Detractor definition: NPS 0-6. Passives (7-8) and Promoters (9-10) are combined into the negative class (detractor_flag = 0).
- Leakage prevention: annual_cltv, risk_score, nps_score, and nps_category are excluded from model training. monthly_revenue is excluded as a proxy leakage risk.
- Train/test split: 80/20 stratified on detractor_flag.
- Model: GradientBoostingClassifier with default hyperparameters; no grid search applied in v1.0.

**Business assumptions:**
- CLTV definition: monthly_revenue x 12 months, adjusted for tenure. Represents forward 12-month value per account.
- Recovery program outreach cost: $25 per contact (default in financial simulator; adjustable).
- Save rate default: 20% (adjustable in simulator). Net value = (total Detractor CLTV x save rate) minus total outreach cost.
- Repeat contact threshold: 3+ contacts in 90 days is used as the high-friction flag; this threshold is based on the dataset distribution, not a client-defined SLA.

---

## 👤 Author

**Luciano Casillas**
Independent Analytics Consultant | Austin, TX

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/luciano-casillas)
[![GitHub](https://img.shields.io/badge/GitHub-Aztexan512-lightgrey)](https://aztexan512.github.io/)

luciano.casillasjr@outlook.com
