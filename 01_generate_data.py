"""
NovaStar Telecom VOC Survey Dataset Generator
Author: Luciano Casillas
Version: 1.0

Generates a synthetic 120,000-row Voice of Customer (VOC) survey dataset
for NovaStar Telecom. Each row represents one post-interaction survey response
linked to a customer account. The dataset models realistic NPS distributions,
satisfaction driver scores, and behavioral signals associated with Detractor outcomes.

Output:
    data/voc_survey.csv         -- Primary survey dataset
    data/voc_metadata.json      -- Generation metadata
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta

# =============================================================
# CONFIGURATION
# =============================================================
N_ROWS   = 120_000
SEED     = 42
VERSION  = "1.0"
OUT_CSV  = os.path.join(os.path.dirname(__file__), "data", "voc_survey.csv")
OUT_META = os.path.join(os.path.dirname(__file__), "data", "voc_metadata.json")

rng = np.random.default_rng(SEED)

# =============================================================
# SEGMENT DEFINITIONS
# =============================================================
CUSTOMER_SEGMENTS = ["Consumer", "Small Business", "Enterprise"]
SEG_WEIGHTS       = [0.62, 0.27, 0.11]

SERVICE_TYPES = ["Mobile", "Broadband", "Bundle", "TV Only"]
SVC_WEIGHTS   = [0.38, 0.28, 0.24, 0.10]

CHANNELS = ["Call Center", "In-Store", "Chat/Digital", "Self-Serve App", "Field Tech"]
CHAN_WEIGHTS = [0.31, 0.18, 0.27, 0.17, 0.07]

REGIONS = ["West", "Southwest", "Midwest", "Southeast", "Northeast"]
REG_WEIGHTS = [0.22, 0.18, 0.20, 0.21, 0.19]

PLAN_TIERS = ["Basic", "Standard", "Premium", "Business Plus"]
PLAN_WEIGHTS = [0.22, 0.37, 0.28, 0.13]

INTERACTION_REASONS = [
    "Billing Dispute", "Technical Support", "Service Upgrade",
    "Cancellation Request", "New Activation", "General Inquiry", "Outage Follow-Up"
]
REASON_WEIGHTS = [0.18, 0.24, 0.13, 0.09, 0.11, 0.15, 0.10]

# =============================================================
# DRAW CATEGORICAL COLUMNS
# =============================================================
print("Drawing categorical attributes...")

customer_id   = [f"NST-{i:07d}" for i in range(1, N_ROWS + 1)]
segment       = rng.choice(CUSTOMER_SEGMENTS, size=N_ROWS, p=SEG_WEIGHTS)
service_type  = rng.choice(SERVICE_TYPES,     size=N_ROWS, p=SVC_WEIGHTS)
channel       = rng.choice(CHANNELS,          size=N_ROWS, p=CHAN_WEIGHTS)
region        = rng.choice(REGIONS,           size=N_ROWS, p=REG_WEIGHTS)
plan_tier     = rng.choice(PLAN_TIERS,        size=N_ROWS, p=PLAN_WEIGHTS)
interact_rsn  = rng.choice(INTERACTION_REASONS, size=N_ROWS, p=REASON_WEIGHTS)

# =============================================================
# TENURE AND ACCOUNT AGE
# =============================================================
print("Generating tenure and financial columns...")

# Tenure in months: skewed toward 12-60 months with some long-tenured
tenure_raw = rng.gamma(shape=3.0, scale=18.0, size=N_ROWS)
tenure_months = np.clip(tenure_raw, 1, 144).astype(int)

# Monthly revenue per account (USD) -- varies by plan tier and service type
base_revenue = {"Basic": 35, "Standard": 65, "Business Plus": 120, "Premium": 95}
rev_base = np.array([base_revenue[p] for p in plan_tier], dtype=float)
rev_noise = rng.normal(0, 8, size=N_ROWS)
monthly_revenue = np.clip(rev_base + rev_noise, 15, 250).round(2)

# Estimated annual CLTV (12 months forward value, simplified)
cltv = (monthly_revenue * 12 * (1 + tenure_months / 144 * 0.25)).round(2)

# =============================================================
# SATISFACTION DRIVER SCORES (1-10 scale)
# Drivers: Resolution Ease, Agent Professionalism, Wait Time,
#          First Contact Resolution, Digital Experience, Billing Clarity
# Each driver has a realistic base score with segment/channel modifiers
# =============================================================
print("Generating satisfaction driver scores...")

def draw_driver(base, std=1.6, lo=1, hi=10):
    raw = rng.normal(base, std, size=N_ROWS)
    return np.clip(raw, lo, hi).round(1)

# Base scores are set to produce realistic driver distributions
resolution_ease     = draw_driver(7.1)
agent_prof          = draw_driver(7.8)
wait_time_score     = draw_driver(6.4)   # deliberately lower -- known pain point
first_contact_res   = draw_driver(6.9)
digital_experience  = draw_driver(7.2)
billing_clarity     = draw_driver(7.0)

# Call Center drags wait time and FCR scores
cc_mask = channel == "Call Center"
wait_time_score  = np.where(cc_mask, wait_time_score  - rng.uniform(0.3, 0.9, N_ROWS), wait_time_score)
first_contact_res = np.where(cc_mask, first_contact_res - rng.uniform(0.2, 0.7, N_ROWS), first_contact_res)

# Billing Dispute interactions drag billing clarity
bill_mask = interact_rsn == "Billing Dispute"
billing_clarity = np.where(bill_mask, billing_clarity - rng.uniform(0.5, 1.4, N_ROWS), billing_clarity)

# Technical Support and Outage Follow-Up drag resolution ease
tech_mask = np.isin(interact_rsn, ["Technical Support", "Outage Follow-Up"])
resolution_ease = np.where(tech_mask, resolution_ease - rng.uniform(0.4, 1.2, N_ROWS), resolution_ease)

# Re-clip after modifiers
resolution_ease   = np.clip(resolution_ease, 1, 10).round(1)
wait_time_score   = np.clip(wait_time_score, 1, 10).round(1)
first_contact_res = np.clip(first_contact_res, 1, 10).round(1)
billing_clarity   = np.clip(billing_clarity, 1, 10).round(1)

# =============================================================
# BEHAVIORAL SIGNALS
# =============================================================
print("Generating behavioral signals...")

# Prior contacts in last 90 days (repeat contact = friction signal)
prior_contacts_90d = rng.poisson(lam=1.4, size=N_ROWS).clip(0, 10)
# Cancellation request boosts prior contacts
prior_contacts_90d = np.where(
    interact_rsn == "Cancellation Request",
    prior_contacts_90d + rng.integers(1, 4, N_ROWS),
    prior_contacts_90d
).clip(0, 12)

# Days since last outage (0 = no outage in last 30 days)
days_since_outage = rng.integers(0, 90, size=N_ROWS)
# West and Southwest regions have more recent outages (narrative)
outage_mask = np.isin(region, ["West", "Southwest"])
days_since_outage = np.where(outage_mask, days_since_outage - rng.integers(0, 20, N_ROWS), days_since_outage).clip(0, 90)

# Interaction resolution flag: was the issue resolved in this interaction?
# Modeled as a function of FCR score and interaction reason
fcr_prob = np.clip((first_contact_res - 1) / 9.0, 0.05, 0.95)
# Lower resolution probability for escalation-heavy reasons
fcr_adj = np.where(np.isin(interact_rsn, ["Billing Dispute", "Cancellation Request", "Technical Support"]), fcr_prob - 0.15, fcr_prob)
resolved_flag = (rng.uniform(0, 1, N_ROWS) < np.clip(fcr_adj, 0.05, 0.95)).astype(int)

# Number of escalations in interaction (0-3)
escalation_count = np.where(
    resolved_flag == 0,
    rng.integers(1, 4, N_ROWS),
    rng.integers(0, 2, N_ROWS)
).clip(0, 3)

# =============================================================
# SURVEY RESPONSE METADATA
# =============================================================
print("Generating survey metadata...")

# Survey dates: last 24 months
base_date = datetime(2023, 4, 1)
date_offsets = rng.integers(0, 730, size=N_ROWS)
survey_dates = [base_date + timedelta(days=int(d)) for d in date_offsets]
survey_date_str = [d.strftime("%Y-%m-%d") for d in survey_dates]

# Verbatim response provided flag (higher for detractors -- they talk more)
verbatim_flag = rng.binomial(1, 0.38, N_ROWS)  # will adjust post-NPS

# =============================================================
# TARGET VARIABLE: NPS SCORE AND DETRACTOR FLAG
# Detractor = NPS 0-6, Passive = 7-8, Promoter = 9-10
# Modeled as a weighted function of driver scores + behavioral signals
# =============================================================
print("Modeling NPS scores and Detractor target...")

# Weighted composite satisfaction score
composite = (
    0.28 * resolution_ease +
    0.18 * agent_prof +
    0.22 * wait_time_score +
    0.20 * first_contact_res +
    0.07 * digital_experience +
    0.05 * billing_clarity
)

# Adjust composite for behavioral signals
composite -= 0.15 * np.log1p(prior_contacts_90d)
composite -= 0.08 * (days_since_outage < 7).astype(float) * 2.0
composite -= 0.20 * (resolved_flag == 0).astype(float) * 1.5
composite -= 0.10 * escalation_count

# Normalize composite to NPS probability space
comp_norm = (composite - composite.mean()) / composite.std()

# Draw NPS 0-10 using a shifted Beta -> map to integers
# Detractor probability is a logistic function of comp_norm
detractor_prob = 1 / (1 + np.exp(2.0 * comp_norm + 0.4))
passive_prob   = np.clip(0.22 * np.ones(N_ROWS) - 0.04 * comp_norm, 0.10, 0.30)
promoter_prob  = np.clip(1 - detractor_prob - passive_prob, 0.05, 0.85)

# Draw NPS category, then score within category
nps_category = np.array([
    rng.choice(["Detractor", "Passive", "Promoter"],
               p=[d, pa, pr] / np.array([d, pa, pr]).sum())
    for d, pa, pr in zip(detractor_prob, passive_prob, promoter_prob)
])

nps_score = np.where(
    nps_category == "Detractor",
    rng.integers(0, 7, N_ROWS),
    np.where(nps_category == "Passive", rng.integers(7, 9, N_ROWS), rng.integers(9, 11, N_ROWS))
)

detractor_flag = (nps_category == "Detractor").astype(int)

# Verbatim more likely from detractors
verbatim_flag = np.where(
    detractor_flag == 1,
    rng.binomial(1, 0.58, N_ROWS),
    rng.binomial(1, 0.28, N_ROWS)
)

# Derived risk score (dashboard only -- not for ML training)
# Combines NPS score inversion + prior contacts + unresolved flag
risk_score = (
    (10 - nps_score) * 0.40 +
    np.clip(prior_contacts_90d, 0, 5) * 0.30 +
    (1 - resolved_flag) * 2.5 +
    escalation_count * 0.40
)
risk_score = np.clip(risk_score, 0, 10).round(2)

# =============================================================
# ASSEMBLE DATAFRAME
# =============================================================
print("Assembling DataFrame...")

df = pd.DataFrame({
    "customer_id":         customer_id,
    "survey_date":         survey_date_str,
    "customer_segment":    segment,
    "service_type":        service_type,
    "interaction_channel": channel,
    "interaction_reason":  interact_rsn,
    "region":              region,
    "plan_tier":           plan_tier,
    "tenure_months":       tenure_months,
    "monthly_revenue":     monthly_revenue,
    "annual_cltv":         cltv,
    "resolution_ease":     resolution_ease,
    "agent_professionalism": agent_prof,
    "wait_time_score":     wait_time_score,
    "first_contact_resolution": first_contact_res,
    "digital_experience":  digital_experience,
    "billing_clarity":     billing_clarity,
    "prior_contacts_90d":  prior_contacts_90d,
    "days_since_outage":   days_since_outage,
    "resolved_flag":       resolved_flag,
    "escalation_count":    escalation_count,
    "verbatim_provided":   verbatim_flag,
    "nps_score":           nps_score,
    "nps_category":        nps_category,
    "detractor_flag":      detractor_flag,   # TARGET VARIABLE
    "risk_score":          risk_score,       # DASHBOARD ONLY -- not for ML
})

# =============================================================
# VALIDATION
# =============================================================
print("Running validation checks...")
assert df.shape[0] == N_ROWS, "Row count mismatch"
assert set(df["detractor_flag"].unique()).issubset({0, 1}), "Target not binary"
assert df[["customer_id", "tenure_months", "monthly_revenue", "detractor_flag"]].isnull().sum().sum() == 0
print(f"  Detractor rate: {df['detractor_flag'].mean():.3f}")
print(f"  NPS distribution: {df['nps_category'].value_counts().to_dict()}")
print(f"  Rows: {df.shape[0]:,}  Cols: {df.shape[1]}")

# =============================================================
# SAVE OUTPUTS
# =============================================================
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"CSV saved: {OUT_CSV}")

metadata = {
    "project":         "NovaStar Telecom VOC Survey Analysis",
    "version":         VERSION,
    "seed":            SEED,
    "n_rows":          N_ROWS,
    "n_cols":          df.shape[1],
    "target_variable": "detractor_flag",
    "target_rate":     round(df["detractor_flag"].mean(), 4),
    "generated_at":    datetime.utcnow().isoformat() + "Z",
    "columns":         list(df.columns),
}
with open(OUT_META, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved: {OUT_META}")
print("Done.")
