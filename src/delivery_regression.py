import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
from load_data import load_data

np.random.seed(42)

# Load the data
df = load_data()
print(f"Total orders available: {len(df):,}")

# Check for missing values
model_columns = [
    "delivery_days", "order_month", "order_year",
    "customer_state", "seller_state", "customer_zip_code_prefix",
    "product_weight_g", "product_length_cm",
    "product_height_cm", "product_width_cm"
]

print("\nMissing values:")
print(df[model_columns].isna().sum())

model_df = df[model_columns].dropna().copy()

# Split the data 60/20/20
train_df, temp_df = train_test_split(model_df, test_size=0.40, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

# Remove outliers from training data
before = len(train_df)

days_from_est = df.loc[train_df.index, "days_from_est"]
train_df = train_df[days_from_est.abs() < 90].copy()

print("Training outliers removed:", before - len(train_df))

print("\nData split:")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

# Check delivery days
print("\nDelivery days summary:")
print(model_df["delivery_days"].describe())

print("Negative delivery days:", (model_df["delivery_days"] < 0).sum())
print("Orders over 60 days:", (model_df["delivery_days"] > 60).sum())
print("Orders over 90 days:", (model_df["delivery_days"] > 90).sum())

# Variables we are using to predict delivery time
feature_columns = [
    "order_month", "order_year", "customer_state",
    "seller_state", "product_weight_g", "product_length_cm",
    "product_height_cm", "product_width_cm"
]

target_column = "delivery_days"

print("\nPredictors:")
print(feature_columns)

X_train = train_df[feature_columns]
y_train = train_df[target_column]

X_val = val_df[feature_columns]
y_val = val_df[target_column]

X_test = test_df[feature_columns]
y_test = test_df[target_column]

print("\nTraining data shape:")
print(X_train.shape)
print(y_train.shape)

print("Unique ZIP codes:", train_df["customer_zip_code_prefix"].nunique())
print("Customer states:", X_train["customer_state"].nunique())
print("Seller states:", X_train["seller_state"].nunique())

# Separate categorical and numerical columns
categorical_columns = ["customer_state", "seller_state", "order_month"]

numerical_columns = [
    "order_year", "product_weight_g", "product_length_cm",
    "product_height_cm", "product_width_cm"
]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ("num", "passthrough", numerical_columns)
])

# Fit using training data only
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print("\nData shape after encoding:")
print("Training:", X_train_processed.shape)
print("Validation:", X_val_processed.shape)
print("Testing:", X_test_processed.shape)


# Linear regression
linear_model = LinearRegression()
linear_model.fit(X_train_processed, y_train)

linear_val_predictions = linear_model.predict(X_val_processed)

linear_mae = mean_absolute_error(y_val, linear_val_predictions)
linear_rmse = np.sqrt(mean_squared_error(y_val, linear_val_predictions))
linear_r2 = r2_score(y_val, linear_val_predictions)

print("\nLinear Regression Results:")
print(f"MAE: {linear_mae:.2f} days")
print(f"RMSE: {linear_rmse:.2f} days")
print(f"R-squared: {linear_r2:.4f}")


# Random forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_processed, y_train)

rf_val_predictions = rf_model.predict(X_val_processed)

rf_mae = mean_absolute_error(y_val, rf_val_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_val_predictions))
rf_r2 = r2_score(y_val, rf_val_predictions)

print("\nRandom Forest Results:")
print(f"MAE: {rf_mae:.2f} days")
print(f"RMSE: {rf_rmse:.2f} days")
print(f"R-squared: {rf_r2:.4f}")

# Compare both models
model_results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [linear_mae, rf_mae],
    "RMSE": [linear_rmse, rf_rmse],
    "R-squared": [linear_r2, rf_r2]
})

print("\nModel Comparison:")
print(model_results.round(4).to_string(index=False))


# Check for repeated orders
if "order_id" in df.columns:
    print("\nOrder ID Check:")
    print("Total rows:", len(df))
    print("Unique orders:", df["order_id"].nunique())
    print("Repeated order IDs:", df["order_id"].duplicated().sum())
else:
    print("Order ID is not available in the dataset.")


# Test random forest on the testing data
rf_test_predictions = rf_model.predict(X_test_processed)

rf_test_mae = mean_absolute_error(y_test, rf_test_predictions)
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_predictions))
rf_test_r2 = r2_score(y_test, rf_test_predictions)

print("\nFinal Random Forest Testing Results:")
print(f"MAE: {rf_test_mae:.2f} days")
print(f"RMSE: {rf_test_rmse:.2f} days")
print(f"R-squared: {rf_test_r2:.4f}")


# Compare with Olist's original delivery estimates
olist_estimates = df.loc[test_df.index, "est_delivery_days"]

olist_mae = mean_absolute_error(y_test, olist_estimates)
olist_rmse = np.sqrt(mean_squared_error(y_test, olist_estimates))

mae_improvement = ((olist_mae - rf_test_mae) / olist_mae) * 100

print("\nOlist Baseline Results:")
print(f"Original MAE: {olist_mae:.2f} days")
print(f"Original RMSE: {olist_rmse:.2f} days")

print("\nDelivery Estimate Improvement:")
print(f"Olist MAE: {olist_mae:.2f} days")
print(f"Random Forest MAE: {rf_test_mae:.2f} days")
print(f"MAE Improvement: {mae_improvement:.2f}%")

if mae_improvement >= 25:
    print("The 25% MAE improvement benchmark was reached.")
else:
    print("The 25% MAE improvement benchmark was not reached.")


# Save results in the outputs folder
output_dir = Path(__file__).resolve().parent.parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

model_results.to_csv(output_dir / "model_comparison.csv", index=False)

final_results = pd.DataFrame({
    "Model": ["Original Olist Estimates", "Random Forest without Buffer"],
    "MAE": [olist_mae, rf_test_mae],
    "RMSE": [olist_rmse, rf_test_rmse],
    "R-squared": [r2_score(y_test, olist_estimates), rf_test_r2]
})

final_results.to_csv(output_dir / "final_model_results.csv", index=False)

improvement_results = pd.DataFrame({
    "MAE Benchmark (%)": [25],
    "Unbuffered MAE Improvement (%)": [mae_improvement],
    "MAE Reduction (Days)": [olist_mae - rf_test_mae],
    "MAE Benchmark Achieved": [mae_improvement >= 25]
})

improvement_results.to_csv(output_dir / "delivery_improvement.csv", index=False)

print("\nModel results saved to the outputs folder.")

# Check Olist's original delivery performance
olist_ratio = (olist_estimates / y_test).median()
olist_late_rate = (y_test > olist_estimates).mean() * 100

print("\nOriginal Delivery Performance:")
print(f"Median delivery ratio: {olist_ratio:.2f}")
print(f"Late deliveries: {olist_late_rate:.2f}%")

# Try different buffer days using validation data
print("\nBuffer Day Comparison:")

for buffer in range(21):
    estimates = np.ceil(rf_val_predictions + buffer)
    ratio = (estimates / y_val).median()
    late_rate = (y_val > estimates).mean() * 100

    print(f"Buffer: {buffer} days, Ratio: {ratio:.2f}, Late: {late_rate:.2f}%")

    # Try buffer days with linear regression
print("\nLinear Regression Buffer Comparison:")

for buffer in range(21):
    estimates = np.ceil(linear_val_predictions + buffer)
    ratio = (estimates / y_val).median()
    late_rate = (y_val > estimates).mean() * 100

    print(f"Buffer: {buffer} days, Ratio: {ratio:.2f}, Late: {late_rate:.2f}%")

    # Save buffer comparison results
buffer_results = []

for buffer in range(21):
    rf_estimates = np.ceil(rf_val_predictions + buffer)
    linear_estimates = np.ceil(linear_val_predictions + buffer)

    buffer_results.append({
        "Buffer Days": buffer,
        "RF Ratio": (rf_estimates / y_val).median(),
        "RF Late (%)": (y_val > rf_estimates).mean() * 100,
        "Linear Ratio": (linear_estimates / y_val).median(),
        "Linear Late (%)": (y_val > linear_estimates).mean() * 100
    })

buffer_df = pd.DataFrame(buffer_results)
buffer_df.to_csv(output_dir / "buffer_comparison.csv", index=False)

print("\nBuffer comparison saved.")

# Test the selected 10-day buffer
final_buffer = 10
final_estimates = np.ceil(rf_test_predictions + final_buffer)

final_ratio = (final_estimates / y_test).median()
final_late_rate = (y_test > final_estimates).mean() * 100
final_mae = mean_absolute_error(y_test, final_estimates)

print("\nFinal Delivery Performance:")
print(f"Buffer days: {final_buffer}")
print(f"Median delivery ratio: {final_ratio:.2f}")
print(f"Late deliveries: {final_late_rate:.2f}%")
print(f"MAE with buffer: {final_mae:.2f} days")

# Save final delivery performance
delivery_results = pd.DataFrame({
    "Model": ["Original Olist", "Random Forest with 10-Day Buffer"],
    "Median Delivery Ratio": [olist_ratio, final_ratio],
    "Late Deliveries (%)": [olist_late_rate, final_late_rate],
    "MAE": [olist_mae, final_mae]
})

delivery_results.to_csv(
    output_dir / "final_delivery_performance.csv",
    index=False
)

print("\nFinal delivery performance saved.")