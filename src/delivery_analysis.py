import pandas as pd
from pathlib import Path
from load_data import load_data

# Load the data
df = load_data()

output_dir = Path(__file__).resolve().parent.parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# Calculate delivery errors
df["absolute_error"] = df["days_from_est"].abs()
df["late_delivery"] = df["days_from_est"] > 0
df["delivery_ratio"] = df["est_delivery_days"] / df["delivery_days"]

# Compare seller and customer states
state_results = df.groupby(
    ["seller_state", "customer_state"]
).agg(
    total_orders=("order_id", "count"),
    average_error=("days_from_est", "mean"),
    mae=("absolute_error", "mean"),
    late_rate=("late_delivery", "mean"),
    median_ratio=("delivery_ratio", "median")
).reset_index()

state_results["late_rate"] = state_results["late_rate"] * 100

# Focus on state combinations with at least 100 orders
state_results = state_results[
    state_results["total_orders"] >= 100
].copy()

state_results = state_results.sort_values("mae", ascending=False)

print("\nState Combinations with Highest Delivery Errors:")
print(state_results.head(10).round(2).to_string(index=False))

# Save geographic results
state_results.to_csv(
    output_dir / "geographic_analysis.csv",
    index=False
)

print("\nGeographic analysis saved.")

# Compare delivery performance by month
monthly_results = df.groupby("order_month").agg(
    total_orders=("order_id", "count"),
    average_error=("days_from_est", "mean"),
    mae=("absolute_error", "mean"),
    late_rate=("late_delivery", "mean"),
    median_ratio=("delivery_ratio", "median")
).reset_index()

monthly_results["late_rate"] = monthly_results["late_rate"] * 100

print("\nMonthly Delivery Performance:")
print(monthly_results.round(2).to_string(index=False))

# Save monthly results
monthly_results.to_csv(
    output_dir / "seasonal_analysis.csv",
    index=False
)

print("\nSeasonal analysis saved.")

# Load multi-item orders
multi_sql = Path(__file__).resolve().parent.parent / "sql" / "multi_item_orders.sql"
multi_df = load_data(multi_sql)

# Check for duplicate orders
print("\nMulti-Item Order Check:")
print("Total orders:", len(multi_df))
print("Repeated order IDs:", multi_df["order_id"].duplicated().sum())

# Calculate delivery errors
multi_df["absolute_error"] = multi_df["days_from_est"].abs()
multi_df["late_delivery"] = multi_df["days_from_est"] > 0
multi_df["delivery_ratio"] = multi_df["est_delivery_days"] / multi_df["delivery_days"]

# Compare single-item and multi-item orders
item_results = pd.DataFrame({
    "Order Type": ["Single Item", "Multiple Items"],
    "Total Orders": [len(df), len(multi_df)],
    "Average Error": [
        df["days_from_est"].mean(),
        multi_df["days_from_est"].mean()
    ],
    "MAE": [
        df["absolute_error"].mean(),
        multi_df["absolute_error"].mean()
    ],
    "Late Rate (%)": [
        df["late_delivery"].mean() * 100,
        multi_df["late_delivery"].mean() * 100
    ],
    "Median Ratio": [
        df["delivery_ratio"].median(),
        multi_df["delivery_ratio"].median()
    ]
})

print("\nSingle-Item vs Multi-Item Orders:")
print(item_results.round(2).to_string(index=False))

# Save item comparison
item_results.to_csv(
    output_dir / "multi_item_analysis.csv",
    index=False
)

print("\nMulti-item analysis saved.")