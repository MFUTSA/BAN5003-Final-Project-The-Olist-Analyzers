import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate the outputs folder
output_dir = Path(__file__).resolve().parent.parent / "outputs"

# Load saved results
models = pd.read_csv(output_dir / "model_comparison.csv")
buffers = pd.read_csv(output_dir / "buffer_comparison.csv")
states = pd.read_csv(output_dir / "geographic_analysis.csv")
months = pd.read_csv(output_dir / "seasonal_analysis.csv")
final = pd.read_csv(output_dir / "final_delivery_performance.csv")

# Graph 1: Model comparison
plt.figure(figsize=(8, 5))
bars = plt.bar(models["Model"], models["MAE"])
plt.bar_label(bars, fmt="%.2f", padding=3)
plt.ylim(0, 6)
plt.title("Model Accuracy Comparison - Validation Data")
plt.ylabel("Mean Absolute Error (Days)")
plt.tight_layout()
plt.savefig(output_dir / "model_comparison.png", dpi=300)
plt.close()

# Graph 2: Buffer comparison
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(buffers["Buffer Days"], buffers["RF Ratio"],
         color="blue", label="Delivery Ratio")
ax1.set_xlabel("Buffer Days")
ax1.set_ylabel("Median Delivery Ratio", color="blue")

ax1.axhspan(1.00, 1.25, color="blue", alpha=0.10,
            label="Target Ratio: 1.00-1.25")
ax1.set_ylim(0.9, 3.5)

ax2 = ax1.twinx()
ax2.plot(buffers["Buffer Days"], buffers["RF Late (%)"],
         color="red", label="Late Delivery Rate")
ax2.set_ylabel("Late Deliveries (%)", color="red")
ax2.axhline(6.9, color="red", linestyle="--",
            label="6.9% Late Threshold")

ax1.axvline(10, color="gray", linestyle="--",
            label="Selected 10-Day Buffer")

ax1.legend(loc="upper center", fontsize=8)
ax2.legend(loc="upper right", fontsize=8)

plt.title("Random Forest Buffer Trade-Off - Validation Data")
fig.tight_layout()
plt.savefig(output_dir / "buffer_comparison.png", dpi=300)
plt.close()

# Graph 3: Geographic analysis
top_states = states.head(10).copy()
top_states["route"] = (
    top_states["seller_state"] + " to " +
    top_states["customer_state"]
)

plt.figure(figsize=(10, 6))
bars = plt.barh(top_states["route"], top_states["mae"])
plt.bar_label(bars, fmt="%.2f", padding=3)
plt.gca().invert_yaxis()
plt.xlim(0, 23)
plt.title("Top 10 Routes with Highest Original Delivery Errors\n"
          "Routes with at Least 100 Orders")
plt.xlabel("Mean Absolute Error (Days)")
plt.ylabel("Seller to Customer State")
plt.tight_layout()
plt.savefig(output_dir / "geographic_analysis.png", dpi=300)
plt.close()

# Graph 4: Seasonal analysis
month_names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

plt.figure(figsize=(10, 5))
plt.plot(months["order_month"], months["late_rate"], marker="o")

for x, y in zip(months["order_month"], months["late_rate"]):
    plt.text(x, y + 0.4, f"{y:.2f}%", ha="center", fontsize=9)

plt.axhline(6.9, color="red", linestyle="--",
            label="6.9% Late Threshold")
plt.xticks(range(1, 13), month_names)
plt.ylim(0, 18)
plt.title("Olist's Original Late Delivery Rate by Month")
plt.ylabel("Late Deliveries (%)")
plt.xlabel("Month")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "seasonal_analysis.png", dpi=300)
plt.close()

# Graph 5: Final delivery performance
plt.figure(figsize=(10, 5))
bars = plt.bar(final["Model"], final["Late Deliveries (%)"])
plt.bar_label(bars, fmt="%.2f%%", padding=3)
plt.axhline(6.9, color="red", linestyle="--",
            label="6.9% Late Threshold")
plt.ylim(0, 8.5)
plt.title("Late Delivery Performance - Testing Data")
plt.ylabel("Late Deliveries (%)")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "final_delivery_performance.png", dpi=300)
plt.close()

print("All five visualizations saved.")