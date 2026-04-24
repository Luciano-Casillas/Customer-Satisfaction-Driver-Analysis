-- ============================================================
-- NovaStar Telecom VOC Survey -- SQL Analysis
-- Author: Luciano Casillas
-- Version: 1.0
-- Compatible: PostgreSQL, BigQuery, Snowflake (ANSI SQL)
-- Table: voc_survey
-- ============================================================
-- This file contains 12 queries across 5 analytical sections.
-- Each query answers a specific business question a Customer
-- Insights Data Analyst at a VOC analytics firm would run.
-- ============================================================


-- ============================================================
-- SECTION 1: DATA QUALITY AND OVERVIEW
-- ============================================================

-- Q1: Row count, column-level completeness, and target distribution
-- Why it matters: Confirms data integrity before analysis begins.
-- A VOC dataset with significant missing driver scores would
-- invalidate importance modeling.
SELECT
    COUNT(*)                                              AS total_responses,
    SUM(CASE WHEN customer_segment IS NULL THEN 1 END)   AS null_segment,
    SUM(CASE WHEN service_type IS NULL THEN 1 END)       AS null_service_type,
    SUM(CASE WHEN resolution_ease IS NULL THEN 1 END)    AS null_resolution_ease,
    SUM(CASE WHEN detractor_flag IS NULL THEN 1 END)     AS null_target,
    SUM(detractor_flag)                                  AS total_detractors,
    ROUND(AVG(detractor_flag) * 100, 2)                  AS detractor_rate_pct,
    ROUND(
        SUM(CASE WHEN nps_category = 'Promoter' THEN 1 END) * 100.0 / COUNT(*), 2
    )                                                    AS promoter_rate_pct,
    ROUND(AVG(nps_score), 2)                             AS avg_nps_score,
    MIN(survey_date)                                     AS earliest_survey,
    MAX(survey_date)                                     AS latest_survey
FROM voc_survey;


-- Q2: NPS category breakdown with CLTV at stake by category
-- Why it matters: Quantifies the revenue exposure associated with
-- each NPS segment. Detractors are more likely to churn and less
-- likely to expand, making their CLTV particularly valuable to protect.
SELECT
    nps_category,
    COUNT(*)                                          AS response_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct_of_total,
    ROUND(AVG(nps_score), 2)                          AS avg_nps_score,
    ROUND(AVG(annual_cltv), 2)                        AS avg_annual_cltv,
    ROUND(SUM(annual_cltv), 0)                        AS total_cltv_at_stake,
    ROUND(AVG(prior_contacts_90d), 2)                 AS avg_prior_contacts
FROM voc_survey
GROUP BY nps_category
ORDER BY avg_nps_score;


-- ============================================================
-- SECTION 2: SEGMENTATION ANALYSIS
-- ============================================================

-- Q3: Detractor rate by every primary categorical dimension
-- Why it matters: Identifies which segment combinations drive
-- the highest complaint concentrations. Informs where to focus
-- CX improvement initiatives first.
WITH segment_rates AS (
    SELECT 'Customer Segment' AS dimension, customer_segment AS segment_value,
           COUNT(*) AS responses, ROUND(AVG(detractor_flag) * 100, 2) AS detractor_rate,
           ROUND(SUM(annual_cltv), 0) AS total_cltv
    FROM voc_survey GROUP BY customer_segment

    UNION ALL

    SELECT 'Service Type', service_type,
           COUNT(*), ROUND(AVG(detractor_flag) * 100, 2), ROUND(SUM(annual_cltv), 0)
    FROM voc_survey GROUP BY service_type

    UNION ALL

    SELECT 'Interaction Channel', interaction_channel,
           COUNT(*), ROUND(AVG(detractor_flag) * 100, 2), ROUND(SUM(annual_cltv), 0)
    FROM voc_survey GROUP BY interaction_channel

    UNION ALL

    SELECT 'Interaction Reason', interaction_reason,
           COUNT(*), ROUND(AVG(detractor_flag) * 100, 2), ROUND(SUM(annual_cltv), 0)
    FROM voc_survey GROUP BY interaction_reason

    UNION ALL

    SELECT 'Region', region,
           COUNT(*), ROUND(AVG(detractor_flag) * 100, 2), ROUND(SUM(annual_cltv), 0)
    FROM voc_survey GROUP BY region

    UNION ALL

    SELECT 'Plan Tier', plan_tier,
           COUNT(*), ROUND(AVG(detractor_flag) * 100, 2), ROUND(SUM(annual_cltv), 0)
    FROM voc_survey GROUP BY plan_tier
)
SELECT
    dimension,
    segment_value,
    responses,
    detractor_rate,
    total_cltv,
    RANK() OVER (PARTITION BY dimension ORDER BY detractor_rate DESC) AS rank_within_dimension
FROM segment_rates
ORDER BY dimension, rank_within_dimension;


-- Q4: Top interaction reason + channel combinations by Detractor rate
-- Why it matters: The cross of "why" and "where" reveals which
-- specific interaction scenarios are failing most severely.
-- Prioritizes process redesign efforts at the intersection level.
SELECT
    interaction_reason,
    interaction_channel,
    COUNT(*)                                           AS responses,
    ROUND(AVG(detractor_flag) * 100, 2)               AS detractor_rate,
    ROUND(AVG(resolution_ease), 2)                    AS avg_resolution_ease,
    ROUND(AVG(wait_time_score), 2)                    AS avg_wait_time,
    ROUND(AVG(first_contact_resolution), 2)           AS avg_fcr_score,
    ROUND(SUM(annual_cltv) / 1000000.0, 2)            AS total_cltv_m
FROM voc_survey
GROUP BY interaction_reason, interaction_channel
HAVING COUNT(*) >= 50
ORDER BY detractor_rate DESC
LIMIT 20;


-- ============================================================
-- SECTION 3: FINANCIAL IMPACT QUANTIFICATION
-- ============================================================

-- Q5: Total CLTV at risk among Detractors by segment and plan tier
-- Why it matters: Frames the business case for CX investment.
-- The CLTV at risk is the upper bound of revenue protected if
-- the Detractor is retained and their NPS improves.
WITH detractor_base AS (
    SELECT
        customer_segment,
        plan_tier,
        COUNT(*)                                         AS detractor_count,
        ROUND(SUM(annual_cltv), 0)                       AS detractor_cltv,
        ROUND(AVG(annual_cltv), 2)                       AS avg_detractor_cltv,
        ROUND(AVG(nps_score), 2)                         AS avg_nps_score
    FROM voc_survey
    WHERE detractor_flag = 1
    GROUP BY customer_segment, plan_tier
)
SELECT
    customer_segment,
    plan_tier,
    detractor_count,
    detractor_cltv,
    avg_detractor_cltv,
    avg_nps_score,
    ROUND(detractor_cltv * 100.0 / SUM(detractor_cltv) OVER(), 2) AS pct_of_total_risk
FROM detractor_base
ORDER BY detractor_cltv DESC;


-- Q6: Simulated recovery value at different save rates
-- Why it matters: Translates the CLTV risk figure into a
-- practical ROI model for a CX intervention campaign.
-- A 20% save rate on top-risk Detractors yields a concrete
-- revenue protection target that executives can approve.
WITH detractor_cltv AS (
    SELECT SUM(annual_cltv) AS total_detractor_cltv
    FROM voc_survey
    WHERE detractor_flag = 1
)
SELECT
    save_rate_pct,
    ROUND(total_detractor_cltv * save_rate_pct / 100.0, 0)         AS revenue_protected,
    ROUND(total_detractor_cltv * save_rate_pct / 100.0 / 1000000, 2) AS revenue_protected_m
FROM detractor_cltv
CROSS JOIN (
    VALUES (5), (10), (15), (20), (25), (30)
) AS scenarios(save_rate_pct)
ORDER BY save_rate_pct;


-- ============================================================
-- SECTION 4: DRIVER AND BEHAVIORAL ANALYSIS
-- ============================================================

-- Q7: Average driver scores by NPS category
-- Why it matters: This is the core importance-performance table.
-- Comparing mean driver scores across Detractor, Passive, and Promoter
-- populations reveals which attributes have the widest gaps --
-- indicating where improvements will generate the most NPS lift.
SELECT
    nps_category,
    COUNT(*)                                       AS n,
    ROUND(AVG(resolution_ease), 2)                 AS avg_resolution_ease,
    ROUND(AVG(agent_professionalism), 2)           AS avg_agent_prof,
    ROUND(AVG(wait_time_score), 2)                 AS avg_wait_time,
    ROUND(AVG(first_contact_resolution), 2)        AS avg_fcr,
    ROUND(AVG(digital_experience), 2)             AS avg_digital_exp,
    ROUND(AVG(billing_clarity), 2)                AS avg_billing_clarity,
    ROUND(AVG(resolution_ease + wait_time_score +
              first_contact_resolution) / 3.0, 2)  AS avg_top3_driver_composite
FROM voc_survey
GROUP BY nps_category
ORDER BY CASE nps_category WHEN 'Detractor' THEN 1 WHEN 'Passive' THEN 2 ELSE 3 END;


-- Q8: Tenure cohort analysis -- Detractor rate by account age bucket
-- Why it matters: Reveals lifecycle risk patterns. New customers
-- often have elevated Detractor rates during onboarding friction.
-- Long-tenured customers who become Detractors represent the
-- highest churn risk because they have no further loyalty buffer.
WITH cohort_buckets AS (
    SELECT
        CASE
            WHEN tenure_months BETWEEN 1  AND 6   THEN '01-06 mo'
            WHEN tenure_months BETWEEN 7  AND 12  THEN '07-12 mo'
            WHEN tenure_months BETWEEN 13 AND 24  THEN '13-24 mo'
            WHEN tenure_months BETWEEN 25 AND 48  THEN '25-48 mo'
            WHEN tenure_months BETWEEN 49 AND 72  THEN '49-72 mo'
            ELSE '73+ mo'
        END AS tenure_bucket,
        tenure_months,
        detractor_flag,
        annual_cltv,
        resolution_ease,
        wait_time_score
    FROM voc_survey
)
SELECT
    tenure_bucket,
    COUNT(*)                                        AS responses,
    ROUND(AVG(detractor_flag) * 100, 2)             AS detractor_rate,
    ROUND(AVG(annual_cltv), 2)                      AS avg_cltv,
    ROUND(AVG(resolution_ease), 2)                  AS avg_resolution_ease,
    ROUND(AVG(wait_time_score), 2)                  AS avg_wait_time
FROM cohort_buckets
GROUP BY tenure_bucket
ORDER BY MIN(tenure_months);


-- Q9: Repeat contact impact on Detractor rate and driver scores
-- Why it matters: Repeat contacts are the primary friction indicator
-- in VOC research. Customers who contact a company 3+ times in 90
-- days are in a failure loop -- each additional contact lowers
-- resolution confidence and raises Detractor probability.
SELECT
    CASE
        WHEN prior_contacts_90d = 0 THEN '0 contacts'
        WHEN prior_contacts_90d = 1 THEN '1 contact'
        WHEN prior_contacts_90d = 2 THEN '2 contacts'
        WHEN prior_contacts_90d BETWEEN 3 AND 5 THEN '3-5 contacts'
        ELSE '6+ contacts'
    END AS contact_frequency_band,
    prior_contacts_90d,
    COUNT(*)                                        AS responses,
    ROUND(AVG(detractor_flag) * 100, 2)             AS detractor_rate,
    ROUND(AVG(resolution_ease), 2)                  AS avg_resolution_ease,
    ROUND(AVG(first_contact_resolution), 2)         AS avg_fcr_score,
    ROUND(AVG(annual_cltv), 2)                      AS avg_cltv
FROM voc_survey
GROUP BY contact_frequency_band, prior_contacts_90d
ORDER BY prior_contacts_90d;


-- ============================================================
-- SECTION 5: MODEL SUPPORT QUERIES
-- ============================================================

-- Q10: Feature engineering -- rolling interaction quality index
-- Why it matters: A composite driver score that normalizes across
-- the five core drivers gives the model (and business analysts)
-- a single "interaction quality" signal that summarizes the
-- survey experience more robustly than any single driver alone.
SELECT
    customer_id,
    survey_date,
    customer_segment,
    interaction_channel,
    interaction_reason,
    detractor_flag,
    -- Weighted composite (weights match model importances)
    ROUND(
        0.28 * resolution_ease +
        0.22 * wait_time_score +
        0.20 * first_contact_resolution +
        0.18 * agent_professionalism +
        0.07 * digital_experience +
        0.05 * billing_clarity,
        2
    ) AS interaction_quality_index,
    resolved_flag,
    escalation_count,
    prior_contacts_90d,
    annual_cltv
FROM voc_survey
ORDER BY interaction_quality_index ASC
LIMIT 1000;


-- Q11: Segment priority ranking for CX intervention
-- Why it matters: Combines Detractor volume, CLTV density, and
-- driver gap (distance from Promoter mean) to produce a
-- prioritization score. Segments with high volume, high value,
-- AND wide driver gaps are highest-ROI intervention targets.
WITH promoter_baseline AS (
    SELECT
        AVG(resolution_ease)          AS p_resolution,
        AVG(wait_time_score)          AS p_wait,
        AVG(first_contact_resolution) AS p_fcr
    FROM voc_survey
    WHERE nps_category = 'Promoter'
),
segment_metrics AS (
    SELECT
        s.customer_segment,
        s.interaction_channel,
        COUNT(*)                                                 AS detractor_count,
        ROUND(AVG(s.detractor_flag) * 100, 2)                   AS detractor_rate,
        ROUND(SUM(s.annual_cltv) / 1000000.0, 2)                AS detractor_cltv_m,
        ROUND(AVG(s.annual_cltv), 2)                            AS avg_cltv,
        ROUND(p.p_resolution - AVG(s.resolution_ease), 2)       AS resolution_gap,
        ROUND(p.p_wait       - AVG(s.wait_time_score), 2)       AS wait_gap,
        ROUND(p.p_fcr        - AVG(s.first_contact_resolution), 2) AS fcr_gap
    FROM voc_survey s, promoter_baseline p
    WHERE s.nps_category = 'Detractor'
    GROUP BY s.customer_segment, s.interaction_channel, p.p_resolution, p.p_wait, p.p_fcr
)
SELECT
    customer_segment,
    interaction_channel,
    detractor_count,
    detractor_rate,
    detractor_cltv_m,
    avg_cltv,
    resolution_gap,
    wait_gap,
    fcr_gap,
    -- Priority score: volume * CLTV density * average driver gap
    ROUND(
        (detractor_count / 100.0) *
        (detractor_cltv_m) *
        ((resolution_gap + wait_gap + fcr_gap) / 3.0 + 0.01),
        2
    ) AS intervention_priority_score
FROM segment_metrics
HAVING detractor_count >= 50
ORDER BY intervention_priority_score DESC
LIMIT 20;


-- Q12: At-risk customer profile -- intervention candidate list
-- Why it matters: This is the operational output of the analysis.
-- It identifies the specific customers in the top-risk deciles
-- who also have high CLTV, ensuring the outreach program spends
-- its budget on the accounts with the highest recovery value.
-- Feed this list to a CRM or contact center for personalized outreach.
WITH scored AS (
    SELECT
        customer_id,
        survey_date,
        customer_segment,
        service_type,
        interaction_channel,
        interaction_reason,
        region,
        plan_tier,
        tenure_months,
        monthly_revenue,
        annual_cltv,
        nps_score,
        nps_category,
        detractor_flag,
        risk_score,
        resolved_flag,
        escalation_count,
        prior_contacts_90d,
        -- Composite driver gap relative to Promoter baseline
        ROUND(
            0.28 * resolution_ease +
            0.22 * wait_time_score +
            0.20 * first_contact_resolution +
            0.18 * agent_professionalism +
            0.07 * digital_experience +
            0.05 * billing_clarity,
            2
        ) AS interaction_quality_index,
        NTILE(10) OVER (ORDER BY risk_score DESC) AS risk_decile
    FROM voc_survey
    WHERE detractor_flag = 1
),
enriched AS (
    SELECT
        *,
        CASE
            WHEN risk_decile = 1 AND annual_cltv > 800  THEN 'Priority 1 -- High Risk, High Value'
            WHEN risk_decile = 1                         THEN 'Priority 2 -- High Risk'
            WHEN risk_decile <= 3 AND annual_cltv > 800  THEN 'Priority 2 -- High Value'
            ELSE 'Standard Outreach'
        END AS outreach_tier
    FROM scored
)
SELECT
    customer_id,
    survey_date,
    customer_segment,
    service_type,
    plan_tier,
    region,
    interaction_channel,
    interaction_reason,
    tenure_months,
    annual_cltv,
    nps_score,
    risk_score,
    risk_decile,
    resolved_flag,
    escalation_count,
    prior_contacts_90d,
    interaction_quality_index,
    outreach_tier
FROM enriched
WHERE risk_decile <= 3
ORDER BY risk_score DESC, annual_cltv DESC
LIMIT 500;

-- ============================================================
-- END OF ANALYSIS FILE
-- ============================================================
