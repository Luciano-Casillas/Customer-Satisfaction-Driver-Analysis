# Data Dictionary -- NovaStar Telecom VOC Survey Dataset
## Project: NovaStar Telecom Customer Experience Analysis
## File: data/voc_survey.csv
## Rows: 120,000 | Columns: 26 | Generated with seed=42

---

## Identifier

| Column | Type | Description | Business Meaning |
|---|---|---|---|
| customer_id | string | Unique respondent ID (format: NST-0000001) | Links survey response to customer account. Not used in modeling. |

---

## Survey and Account Metadata

| Column | Type | Description | Business Meaning |
|---|---|---|---|
| survey_date | date (YYYY-MM-DD) | Date survey was completed | Used for trend analysis and cohort bucketing. Covers April 2023 through April 2025. |
| customer_segment | string | Customer type: Consumer, Small Business, Enterprise | Primary segmentation dimension for CX analysis. |
| service_type | string | Product subscribed: Mobile, Broadband, Bundle, TV Only | Secondary segmentation. Bundle customers have higher CLTV and different driver profiles. |
| interaction_channel | string | Channel where interaction occurred: Call Center, In-Store, Chat/Digital, Self-Serve App, Field Tech | Key driver of wait time and FCR outcomes. Channel performance varies significantly. |
| interaction_reason | string | Primary reason for the interaction: Billing Dispute, Technical Support, Service Upgrade, Cancellation Request, New Activation, General Inquiry, Outage Follow-Up | Drives satisfaction baseline. Billing disputes and cancellation requests correlate strongly with Detractor outcome. |
| region | string | Geographic region: West, Southwest, Midwest, Southeast, Northeast | Used for regional CX benchmarking. West and Southwest show elevated outage recency. |
| plan_tier | string | Pricing tier: Basic, Standard, Premium, Business Plus | Proxy for customer value tier. Higher tiers have higher CLTV but also higher expectations. |

---

## Account and Financial Columns

| Column | Type | Description | Business Meaning | Notes |
|---|---|---|---|---|
| tenure_months | integer | Months since customer activation (1-144) | Longer tenure correlates with higher Promoter rates, but very long-tenured customers show increased Detractor risk after service failures. | Used in model training. |
| monthly_revenue | float | Estimated monthly revenue from this account (USD) | Used to size financial opportunity per segment. | DASHBOARD ONLY for financial impact calculations. Not a model training feature due to potential proxy leakage. |
| annual_cltv | float | Estimated 12-month forward customer lifetime value (USD) | Core financial metric. Computed as monthly_revenue * 12 * tenure adjustment. | **DASHBOARD ONLY -- HIGH leakage risk if used in model.** Derived from revenue and tenure. |

---

## Satisfaction Driver Scores (1-10 scale, 10 = most satisfied)

All driver scores are rated by the survey respondent on a 1-10 scale.
These are the primary model features and CX diagnostic columns.

| Column | Type | Description | Business Meaning | Leakage Risk |
|---|---|---|---|---|
| resolution_ease | float | How easy was it to resolve your issue? (1-10) | Top-ranked driver of NPS outcome (importance: 30.3%). Key lever for CX improvement programs. | LOW -- rated by respondent before NPS score in survey flow. |
| agent_professionalism | float | How professional was the agent or representative? (1-10) | Second-tier driver. Varies by channel; In-Store tends to score higher than Call Center. | LOW |
| wait_time_score | float | Satisfaction with how long you waited (1-10) | Second most important driver (importance: 19.7%). Call Center drag is significant. | LOW |
| first_contact_resolution | float | Was your issue resolved without needing to follow up? (1-10) | Third most important driver (importance: 19.1%). Strongly linked to resolved_flag. | LOW |
| digital_experience | float | Satisfaction with the digital or app experience (1-10) | Lower importance overall but critical for Self-Serve App channel respondents. | LOW |
| billing_clarity | float | Clarity of your bill and charges (1-10) | Disproportionately impactful for Billing Dispute interactions. | LOW |

---

## Behavioral Signals

| Column | Type | Description | Business Meaning | Leakage Risk |
|---|---|---|---|---|
| prior_contacts_90d | integer | Number of contacts with NovaStar in prior 90 days (0-12) | Repeat contact is the primary friction signal. Customers with 3+ contacts in 90 days have significantly elevated Detractor rates. | LOW -- prior-period data, no future information leakage. |
| days_since_outage | integer | Days since a network outage affected the customer's area (0-90; 90 = no recent outage) | Recency of outage experience drives dissatisfaction independent of interaction quality. | LOW |
| resolved_flag | integer (0/1) | 1 = issue was resolved in this interaction; 0 = unresolved | Single strongest binary predictor of unresolved dissatisfaction. Unresolved contacts are 2.3x more likely to produce a Detractor. | LOW |
| escalation_count | integer | Number of escalations during this interaction (0-3) | Escalations signal interaction complexity and agent/process failure. | LOW |
| verbatim_provided | integer (0/1) | 1 = customer submitted a written comment; 0 = did not | Detractors provide verbatim at 2x the rate of Promoters. Useful for text analytics routing. | LOW |

---

## NPS and Target Variable

| Column | Type | Description | Business Meaning | Notes |
|---|---|---|---|---|
| nps_score | integer (0-10) | Raw Net Promoter Score rating | Standard VOC metric. Detractor: 0-6, Passive: 7-8, Promoter: 9-10. | Not used directly in model -- target is detractor_flag. |
| nps_category | string | Detractor, Passive, or Promoter | Segmentation label derived from nps_score. Shown in dashboard overview and trend charts. | Derived from nps_score. |
| **detractor_flag** | **integer (0/1)** | **1 = Detractor (NPS 0-6); 0 = Passive or Promoter** | **Binary target variable for the predictive model.** Dataset detractor rate: 40.6%. | **TARGET VARIABLE** |

---

## Derived / Dashboard-Only Columns

| Column | Type | Description | Business Meaning | Notes |
|---|---|---|---|---|
| risk_score | float (0-10) | Composite at-risk score combining NPS inversion, prior contacts, resolution failure, and escalation count | Provides a continuous severity signal for prioritizing intervention outreach. Higher = more at-risk. | **DASHBOARD ONLY -- HIGH leakage risk.** Incorporates nps_score; must never be used as a model feature. |

---

## Leakage Summary

| Risk Level | Columns |
|---|---|
| HIGH -- exclude from model | annual_cltv, risk_score, nps_score, nps_category |
| MEDIUM -- use with caution | monthly_revenue (proxy for plan tier, which is already in features) |
| LOW -- safe for model training | All driver scores, behavioral signals, tenure_months, customer_segment, service_type, interaction_channel, interaction_reason, region, plan_tier |

---

*Data dictionary version 1.0 -- Luciano Casillas*
