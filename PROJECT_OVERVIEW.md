# NovaStar Telecom VOC Survey Analysis

## Project Summary

NovaStar Telecom surveys customers after every service interaction, but has no systematic way to prioritize which dissatisfied customers to contact first. This project analyzes 120,000 synthetic post-interaction survey responses, models Detractor risk using a Gradient Boosting classifier (AUC 0.81, top-decile lift 1.82x), and identifies the CX drivers and segments that explain the most variance in NPS outcomes. The result is a prioritized recovery playbook with a quantified revenue opportunity exceeding $190M in Detractor CLTV.

---

## Business Problem

A Customer Insights Data Analyst at a VOC analytics firm answers: "Which customers are most likely to leave, why, and what should we do about it?" This project turns that question into a structured analysis for a telecom client, delivering five specific findings:

1. **Which satisfaction drivers predict Detractor status most strongly?**
   Resolution ease is the single most powerful predictor (30.3% of model importance) - Detractors rate it 5.8/10 compared to 8.2/10 for Promoters, a 2.4-point gap. Wait time (19.7%) and first contact resolution (19.1%) round out the top three. Together, these three drivers explain 69% of Detractor risk.

2. **Which channels and interaction types produce the highest Detractor concentrations?**
   Call Center interactions produce the highest Detractor rate of any channel, driven by longer wait times and lower first contact resolution scores. Technical Support, Outage Follow-Up, and Cancellation Request are the highest-risk interaction reasons, each exceeding the 40.6% overall Detractor rate.

3. **How much revenue is at risk, and what is the return on a recovery program?**
   Detractor accounts represent more than $190M in annual customer lifetime value. At a 20% save rate and $25 per contact, a targeted recovery program returns approximately $36.8M in net value after all outreach costs - a strongly positive return even under conservative assumptions.

4. **Which customers should the recovery team contact first?**
   Customers in the top risk decile are 1.82x more likely to be Detractors than average. Contacting just the top 20% of scored customers captures 36.3% of all Detractors - more than double what a random outreach approach would yield. Priority 1 accounts combine high risk score with annual CLTV above $800.

5. **How does this framework translate to adjacent industries?**
   Every satisfaction driver in this model maps directly to a patient experience metric in the HCAHPS survey used by healthcare organizations. The same risk-scoring, driver importance, and recovery simulation approach applies to readmission prevention, discharge planning quality, and staff communication improvement programs.

---

## Key Findings

1. **Resolution ease is the dominant driver.** Feature importance: 30.3%. Detractors score 5.8/10 on resolution ease vs. 8.2/10 for Promoters - a 2.4-point gap. This is the single highest-ROI lever for CX improvement programs.

2. **Call Center drives disproportionate Detractor volume.** Call Center interactions produce the highest Detractor rate among all channels. Wait time and FCR scores are systematically lower for Call Center contacts due to volume and queue dynamics.

3. **Repeat contacts are the clearest friction signal.** Customers with 3+ contacts in 90 days have a Detractor rate more than 2x the rate for first-time contacts. A repeat-contact early warning flag in the CRM would give agents immediate context before the interaction begins.

4. **The model concentrates Detractor risk effectively.** Top decile lift of 1.82x means contacting the top 10% of scored customers recovers nearly twice the share of Detractors compared to a random draw. Top-2 decile cumulative gain of 36.3% means 36% of all Detractors can be identified by contacting only 20% of the scored population.

5. **The recovery program is financially viable at multiple save rate assumptions.** At a 20% save rate and $25 per contact, the program returns approximately $36.8M in net value. The revenue simulator allows stakeholders to model their own assumptions.

---

## Live Dashboard - coming soon

The dashboard is organized across six tabs, each addressing a specific layer of the analysis. Sidebar filters apply across all tabs simultaneously, allowing any view to be scoped by customer segment, channel, region, plan tier, service type, or tenure.

**Overview**
Provides a snapshot of overall NPS performance and how the 40.6% Detractor rate breaks down by interaction reason. This is the starting point for understanding where customer dissatisfaction is most concentrated.

**CX Drivers**
Compares satisfaction scores across six experience drivers between Detractor and Promoter populations, and shows how Detractor risk escalates by channel and repeat contact frequency. Identifies which service dimensions have the widest performance gap and where process improvement would generate the most NPS lift.

**Model + Risk**
Displays the Gradient Boosting model's performance metrics, feature importance rankings, and cumulative gain curve. Includes a risk profile explorer for reviewing the characteristics of customers in the top risk deciles - the prioritized outreach population.

**Financial Impact**
Quantifies the revenue at risk from Detractor accounts by segment and service type, and includes an interactive simulator for modeling recovery program ROI at different save rates and outreach costs. Translates the analytical findings into a dollar-denominated business case.

**Recommendations**
Presents a prioritized action roadmap organized by implementation timeline - Immediate (0-30 days), Short-Term (30-90 days), and Strategic (90+ days). Each recommendation is tied directly to a model finding with supporting evidence.

**Healthcare Application**
Maps every NovaStar satisfaction driver to its equivalent in the HCAHPS patient experience framework and outlines three action levers for a healthcare client. Demonstrates how the same analytical approach transfers to an adjacent industry with minimal adaptation.

---

## Recommendations

Actions are sequenced by implementation timeline and expected business impact.

**Immediate (0-30 Days)**

- **Deploy risk scoring to the Call Center queue.**
  Why: Call Center produces the highest Detractor rate of any channel, driven by wait times and first contact resolution scores that are systematically lower than all other channels. The trained model identifies top-decile customers who are 1.82x more likely to become Detractors than average. Routing these contacts to senior agents before they score low - rather than reacting after the survey comes back - shifts the program from reactive to preventive. No new technology is required; this is an API integration with the existing contact routing system.

- **Launch a closed-loop recovery program for top-risk Detractors.**
  Why: $190M in annual customer lifetime value is held by Detractor accounts. Without a systematic outreach program, the majority of that revenue is at risk of churn with no intervention. The model makes outreach efficient: contacting the top 20% of risk-scored customers recovers 36.3% of all Detractors - more than double what a random calling list would yield. At a 20% save rate and $25 per contact, total outreach costs approximately $1.2M against $38M in revenue protected, producing approximately $36.8M in net value.

**Short-Term (30-90 Days)**

- **Redesign resolution scripts for Billing Dispute interactions.**
  Why: Resolution ease is the single strongest predictor of Detractor status, accounting for 30.3% of total model importance. Billing Dispute interactions drag both resolution ease and billing clarity scores below channel averages, producing one of the highest Detractor rates of any interaction reason. Empowering agents to resolve billing issues on the first contact - without requiring supervisor approval for credits up to $50 - directly closes the 2.4-point resolution ease gap between Detractors and Promoters, the largest improvement lever in the entire model.

- **Add a repeat-contact flag to the CRM.**
  Why: Customers who contacted NovaStar 3 or more times in the prior 90 days show a Detractor rate more than 2x the overall baseline of 40.6%. This friction pattern is visible before the interaction begins, yet agents currently have no way to see it. A CRM flag costs nothing to build once the data is connected and gives every agent immediate context - allowing them to open with acknowledgment rather than discovery, which is the single most effective way to change the emotional tone of a high-friction contact.

**Strategic (90+ Days)**

- **Build a continuous driver importance pipeline.**
  Why: The current model reflects a fixed two-year snapshot of customer behavior. Driver importance rankings will shift as channel mix evolves. Digital experience currently accounts for only 2% of model importance - but Self-Serve App is the second-fastest growing channel in the dataset. A monthly retraining pipeline ensures the model reflects current customer behavior and that CX investment priorities stay aligned with where friction is actually occurring, not where it was two years ago.

---

## Adjacent Framework Portability

The analytical methodology developed for NovaStar - driver importance ranking, risk decile scoring, and closed-loop recovery simulation - is not specific to telecom. Any industry that collects structured customer feedback and tracks financial value per account can apply the same approach with minimal adaptation.

**Healthcare: Patient Experience (HCAHPS)**

The most direct translation is to hospital patient experience analytics. Every NovaStar satisfaction driver maps to a published HCAHPS composite measure:

| Telecom Driver | Healthcare Analogue (HCAHPS) |
|---|---|
| Resolution Ease | Care Coordination Score |
| Wait Time Score | Response to Call Button |
| First Contact Resolution | Discharge Planning Completeness |
| Agent Professionalism | Staff Communication Quality |
| Digital Experience | Patient Portal Usability |
| Billing Clarity | Explanation of Charges |
| Prior Contacts (90d) | 30-Day Readmission Risk |

The business problem is identical: which patients are most at risk of a poor outcome, why, and what intervention should happen first? The same risk-scoring model, applied to HCAHPS survey data, produces a prioritized outreach list for care coordinators - directly reducing preventable readmissions and improving CMS star ratings.

**Additional Industries**

The same framework applies to financial services (customer effort score post-transaction), retail and e-commerce (post-purchase NPS, return rate prediction), and B2B SaaS (renewal risk scoring using product usage and support ticket patterns).

---

## Next Steps

A stakeholder presentation deck is planned as the next deliverable, translating these findings and recommendations into executive-ready slides with visualizations and business impact callouts for each key finding.

---

## Technical Stack

- Language: Python 3.12
- Modeling: scikit-learn (GradientBoostingClassifier)
- Dashboard: Streamlit 1.35, Plotly 5.22
- Data: pandas, numpy
- SQL: ANSI SQL (PostgreSQL/BigQuery/Snowflake-compatible)
- Deployment: Streamlit Community Cloud

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
Synthetic data is generated using `numpy.random.default_rng(42)`. Driver scores are drawn from Normal distributions with segment and channel-specific modifiers that create realistic correlations. The Detractor flag is modeled as a logistic function of a weighted composite driver score plus behavioral signals. This produces a dataset with an AUC of 0.81 - realistic for a survey-based model.

**2. Exploratory analysis**
Driver score gaps between Detractor and Promoter populations were computed for all six drivers. Channel and interaction reason cross-tabs were used to identify high-friction combinations. Repeat contact frequency bands were analyzed for Detractor rate escalation.

**3. Modeling**
Gradient Boosting Classifier trained on 12 features (6 driver scores, 4 behavioral signals, tenure, monthly revenue). Leakage columns excluded: annual_cltv, risk_score, nps_score, nps_category. Train/test split: 80/20 stratified. Validated metrics: AUC 0.81, top-decile lift 1.82x, top-2 decile cumulative gain 36.3%.

**4. Financial impact framework**
Annual Customer Lifetime Value (CLTV) is estimated as 12 months of forward revenue per account adjusted for tenure. Total Detractor CLTV is the recovery opportunity ceiling. The revenue simulator multiplies total Detractor CLTV by a save rate and subtracts contact cost to compute net value.

---

*Project version 1.0 - Luciano Casillas | luciano.casillasjr@outlook.com | github.com/Aztexan512*
