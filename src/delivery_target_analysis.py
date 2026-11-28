import pandas as pd
from pathlib import Path

# Load existing buffer comparison results
output_dir = Path(__file__).resolve().parents[1] / "outputs"
df = pd.read_csv(output_dir / "buffer_comparison.csv")

# Define original and proposed business targets
original_min = 1.00
original_max = 1.25
revised_max = 2.25
late_target = 6.9

results = []

# Evaluate both models across all buffer days
for model, prefix in [
    ("Random Forest", "RF"),
    ("Linear Regression", "Linear")
]:
    for _, row in df.iterrows():
        ratio = row[f"{prefix} Ratio"]
        late = row[f"{prefix} Late (%)"]

        results.append({
            "Model": model,
            "Buffer Days": row["Buffer Days"],
            "Delivery Ratio": ratio,
            "Late Delivery (%)": late,
            "Original Target Met": (
                original_min <= ratio <= original_max
                and late < late_target
            ),
            "Revised Target Met": (
                original_min <= ratio <= revised_max
                and late < late_target
            )
        })

results_df = pd.DataFrame(results)

# Save target evaluation results
results_df.to_csv(
    output_dir / "delivery_target_evaluation.csv",
    index=False
)

# Display strategies meeting the original targets
print("\nOriginal Target Evaluation:")
original = results_df[results_df["Original Target Met"]]
print(original.to_string(index=False) if not original.empty
      else "No strategies meet both original targets.")

# Display strategies meeting the revised targets
print("\nRevised Target Evaluation:")
revised = results_df[results_df["Revised Target Met"]]
print(revised.to_string(index=False) if not revised.empty
      else "No strategies meet both revised targets.")

# Identify minimum buffer needed for each model
print("\nMinimum Buffer to Meet Late Delivery Target:")

for model in results_df["Model"].unique():
    eligible = results_df[
        (results_df["Model"] == model)
        & (results_df["Late Delivery (%)"] < late_target)
    ].sort_values("Buffer Days")

    if not eligible.empty:
        print(eligible.head(1).to_string(index=False))

print("\nResults saved to delivery_target_evaluation.csv")