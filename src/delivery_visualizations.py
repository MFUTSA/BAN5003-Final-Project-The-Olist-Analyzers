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
fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(buffers["Buffer Days"], buffers["RF Ratio"],
         color="blue", label="Delivery Ratio")
ax1.set_xlabel("Buffer Days")
ax1.set_ylabel("Median Delivery Ratio", color="blue")

ax2 = ax1.twinx()
ax2.plot(buffers["Buffer Days"], buffers["RF Late (%)"],
         color="red", label="Late Delivery Rate")
ax2.set_ylabel("Late Deliveries (%)", color="red")
ax2.axhline(6.9, color="red", linestyle="--", alpha=0.5)

ax1.axvline(10, color="gray", linestyle="--")

plt.title("Random Forest Buffer Trade-Off")
fig.tight_layout()
plt.savefig(output_dir / "buffer_comparison.png", dpi=300)
plt.close()

# Graph 3: Geographic analysis
top_states = states.head(10).copy()
top_states["route"] = (
    top_states["seller_state"] + " to " +
    top_states["customer_state"]
)

plt.figure(figsize=(9, 6))
plt.barh(top_states["route"], top_states["mae"])
plt.gca().invert_yaxis()
plt.title("Top 10 Routes with Highest Delivery Errors")
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
plt.axhline(6.9, color="red", linestyle="--", label="6.9% Target")
plt.xticks(range(1, 13), month_names)
plt.title("Late Delivery Rate by Month")
plt.ylabel("Late Deliveries (%)")
plt.xlabel("Month")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "seasonal_analysis.png", dpi=300)
plt.close()

# Graph 5: Final delivery performance
plt.figure(figsize=(9, 5))
plt.bar(final["Model"], final["Late Deliveries (%)"])
plt.axhline(6.9, color="red", linestyle="--", label="6.9% Target")
plt.title("Original vs Recommended Delivery Estimates")
plt.ylabel("Late Deliveries (%)")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "final_delivery_performance.png", dpi=300)
plt.close()

print("All five visualizations saved.")