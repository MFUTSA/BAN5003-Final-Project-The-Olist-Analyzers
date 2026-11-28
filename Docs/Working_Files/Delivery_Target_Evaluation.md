# Delivery Target Evaluation – Phase 2

**Contributor:** Anika Bhuyan  
**Date:** November 27, 2026

## Objective

The purpose of this analysis was to evaluate whether the linear regression and random forest models could meet the original and proposed delivery performance targets using different buffer days.

## Methodology

Using the existing buffer comparison results, I ran a separate Python script to evaluate both models across 0–20 buffer days. The original target was a delivery ratio between 1.00 and 1.25 and a late delivery rate below 6.9%. The proposed revised target increased the maximum delivery ratio to 2.25 while maintaining the original late delivery requirement and minimum ratio.

## Results

| Model | Minimum Buffer to Meet Late Target | Delivery Ratio | Late Delivery Rate |
|---|---:|---:|---:|
| Random Forest | 10 days | 2.22 | 6.10% |
| Linear Regression | 11 days | 2.30 | 6.25% |

Neither model met both original targets. However, random forest with a 10-day buffer was the only tested strategy that satisfied both revised targets.

## Business Recommendations

Based on the results, I recommend considering the random forest model with a 10-day buffer. This approach achieves a late delivery rate that is approximately 0.80 percentage points below the original threshold.

However, increasing the buffer may result in overly conservative delivery estimates. Future analysis could explore adjusting buffer days based on customer location, seasonality, and historical delivery performance instead of applying the same buffer to every order.

## Limitations

This analysis used existing model outputs rather than independently retraining the models because the database password was unavailable in my Codespace. The proposed targets should also be evaluated using held-out test data before implementation.

## AI Assistance Disclosure

ChatGPT assisted with developing the Python script, interpreting the results, and drafting the documentation. I executed the script in GitHub Codespaces and reviewed the generated results.