
# AI Component: Delivery Performance Summary

**AI Tool:** ChatGPT (GPT-6.2)  
**Date:** November 26, 2026

## 1. Structured Prompt

**Role:**  
Act as a business communications assistant.

**Context:**  
My team and I analyzed Olist delivery data to determine whether predictive modeling could improve estimated delivery dates. Here are the results from our Python analysis:

| Metric | Original Olist | Random Forest with 10-Day Buffer |
|---|---:|---:|
| Median delivery ratio | 2.30 | 2.21 |
| Late delivery rate | 6.75% | 6.00% |
| MAE | 13.25 days | 11.90 days |

Our goal was to achieve a median delivery ratio between 1.00 and 1.25 while maintaining a late delivery rate below 6.9%.

**Task:**  
Help me communicate these findings in plain English to a logistics manager who may not have a statistical background. Help me explain what improved and why we did not fully meet our original goal.

**Constraints:**  
Use only the results I provided. Do not perform additional calculations, generate new statistics, or make analytical decisions for me. Refuse to make any claims that cannot be supported by the provided results. I want to review and revise the wording myself before including it in our project.

**Output Format:**  
One short paragraph using simple, professional language.

## 2. Original AI Response

The Random Forest model with a 10-day buffer showed improvements in Olist's delivery estimates. The percentage of late deliveries decreased from 6.75% to 6.00%, while the average absolute difference between estimated and actual delivery times decreased from 13.25 to 11.90 days. Although the model met the late delivery target of below 6.9%, the median delivery ratio of 2.21 remained above the target range of 1.00 to 1.25. These results show progress, but further improvements are needed before the model meets both project goals.

## 3. Student Revision

The Random Forest model with a 10-day buffer improved Olist’s delivery estimation performance. The percentage of late deliveries decreased from 6.75% to 6.00%, while the mean absolute difference between estimated and actual delivery times dropped from 13.25 to 11.90 days. Although the model successfully achieved the project’s late delivery target of below 6.9%, the median delivery ratio remained at 2.21, exceeding the target range of 1.00 to 1.25. Overall, the results demonstrate measurable improvements in delivery accuracy and timeliness; however, additional model adjustments are necessary to meet both project objectives.

## 4. Summary of Edits

The original response was revised to improve clarity, use more precise statistical terminology, and better connect the conclusion to the project's original objectives. All numerical results were retained.

An additional AI review suggested replacing "delivery accuracy and timeliness" with "delivery estimation accuracy" because the model adjusts estimated delivery dates rather than actual shipping speeds.

## 5. Verification

All numerical results were checked against `outputs/final_delivery_performance.csv` on November 26, 2026. The reported model results match the saved Python output when rounded to two decimal places.

The AI was provided with existing statistical results for this communication task and was not asked to generate new numerical results.