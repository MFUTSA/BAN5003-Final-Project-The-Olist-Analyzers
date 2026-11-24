import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from load_data import load_data

# Setting a random seed so our results are reproducible
np.random.seed(42)

# Loading the Olist data from PostgreSQL
df = load_data()

# Confirming the data loaded correctly
print(f"Total orders available: {len(df):,}")

# Checking for missing values in our modeling variables
model_columns = [
    "delivery_days",
    "order_month",
    "order_year",
    "customer_state",
    "seller_state",
    "customer_zip_code_prefix",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]

print("\nMissing values in modeling columns:")
print(df[model_columns].isna().sum())

model_df = df[model_columns].dropna().copy()

# Splitting into 60% training and 40% temporary data
train_df, temp_df = train_test_split(
    model_df,
    test_size=0.40,
    random_state=42
)

# Divide the remaining data into validation and testing
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42
)

print("\nData split:")
print(f"Training: {len(train_df):,} orders")
print(f"Validation: {len(val_df):,} orders")
print(f"Testing: {len(test_df):,} orders")


# Checking the delivery-day values before modeling
print("\nDelivery days summary:")
print(model_df["delivery_days"].describe())

print("\nNegative delivery days:")
print((model_df["delivery_days"] < 0).sum())


# Count unusually long deliveries
print("\nOrders taking more than 60 days:")
print((model_df["delivery_days"] > 60).sum())

print("\nOrders taking more than 90 days:")
print((model_df["delivery_days"] > 90).sum())


# Choose the information we'll use to predict delivery time
feature_columns = [
    "order_month",
    "order_year",
    "customer_state",
    "seller_state",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]

# What we're trying to predict
target_column = "delivery_days"

# Check which variables we're using
print("\nPredictors selected:")
print(feature_columns)


# Separate the inputs from the delivery times
X_train = train_df[feature_columns]
y_train = train_df[target_column]

X_val = val_df[feature_columns]
y_val = val_df[target_column]

X_test = test_df[feature_columns]
y_test = test_df[target_column]

# Check the training data
print("\nTraining data shape:")
print(X_train.shape)
print(y_train.shape)

# See how many ZIP codes are in the training data
print("\nUnique ZIP codes:")
print(train_df["customer_zip_code_prefix"].nunique())

# Check the number of customer and seller states
print("\nCustomer states:")
print(X_train["customer_state"].nunique())

print("\nSeller states:")
print(X_train["seller_state"].nunique())


# Columns that need to be encoded
categorical_columns = [
    "customer_state",
    "seller_state",
    "order_month"
]

# Columns that are already numerical
numerical_columns = [
    "order_year",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]

# Prepare the columns for the regression model
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("num", "passthrough", numerical_columns)
    ]
)

# Fit the preprocessor using only the training data
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same changes to validation and testing
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# Check the data after encoding
print("\nData shape after encoding:")
print("Training:", X_train_processed.shape)
print("Validation:", X_val_processed.shape)
print("Testing:", X_test_processed.shape)

from sklearn.linear_model import LinearRegression

# Start with linear regression
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train_processed, y_train)

# See how well we can predict delivery times
linear_val_predictions = linear_model.predict(X_val_processed)

print("\nLinear Regression model trained successfully.")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Check how close our predictions are to the actual delivery times
linear_mae = mean_absolute_error(y_val, linear_val_predictions)
linear_rmse = np.sqrt(mean_squared_error(y_val, linear_val_predictions))
linear_r2 = r2_score(y_val, linear_val_predictions)

# Print the results
print("\nLinear Regression Results:")
print(f"MAE: {linear_mae:.2f} days")
print(f"RMSE: {linear_rmse:.2f} days")
print(f"R-squared: {linear_r2:.4f}")

from sklearn.ensemble import RandomForestRegressor

# Try random forest to see if it predicts delivery times better
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the random forest model
rf_model.fit(X_train_processed, y_train)

# Predict delivery times using the validation data
rf_val_predictions = rf_model.predict(X_val_processed)

print("\nRandom Forest model trained successfully.")

# Check how accurate the random forest predictions are
rf_mae = mean_absolute_error(y_val, rf_val_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_val_predictions))
rf_r2 = r2_score(y_val, rf_val_predictions)

# Print the random forest results
print("\nRandom Forest Results:")
print(f"MAE: {rf_mae:.2f} days")
print(f"RMSE: {rf_rmse:.2f} days")
print(f"R-squared: {rf_r2:.4f}")

# Put both model results together so we can compare them
model_results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [linear_mae, rf_mae],
    "RMSE": [linear_rmse, rf_rmse],
    "R-squared": [linear_r2, rf_r2]
})

# Display the results side by side
print("\nModel Comparison:")
print(model_results.round(4).to_string(index=False))

# Check if the same order appears more than once
if "order_id" in df.columns:
    print("\nOrder ID Check:")
    print(f"Total rows: {len(df):,}")
    print(f"Unique orders: {df['order_id'].nunique():,}")
    print(f"Repeated order IDs: {df['order_id'].duplicated().sum():,}")
else:
    print("\nOrder ID is not available in the dataset.")

    # Test the random forest model on unseen data
rf_test_predictions = rf_model.predict(X_test_processed)

# Calculate the final model performance
rf_test_mae = mean_absolute_error(y_test, rf_test_predictions)
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_predictions))
rf_test_r2 = r2_score(y_test, rf_test_predictions)

# Display the final testing results
print("\nFinal Random Forest Testing Results:")
print(f"MAE: {rf_test_mae:.2f} days")
print(f"RMSE: {rf_test_rmse:.2f} days")
print(f"R-squared: {rf_test_r2:.4f}")

# Get Olist's original estimates for the same testing orders
olist_estimates = df.loc[test_df.index, "est_delivery_days"]

# Calculate the error in Olist's original estimates
olist_mae = mean_absolute_error(y_test, olist_estimates)
olist_rmse = np.sqrt(mean_squared_error(y_test, olist_estimates))

# Calculate how much our model improves the original estimates
mae_improvement = ((olist_mae - rf_test_mae) / olist_mae) * 100

# Display the comparison results
print("\nOlist Baseline Results:")
print(f"Original MAE: {olist_mae:.2f} days")
print(f"Original RMSE: {olist_rmse:.2f} days")

print("\nDelivery Estimate Improvement:")
print(f"Olist MAE: {olist_mae:.2f} days")
print(f"Random Forest MAE: {rf_test_mae:.2f} days")
print(f"MAE Improvement: {mae_improvement:.2f}%")

# Check if we reached our project goal
if mae_improvement >= 25:
    print("The 25% improvement goal was achieved.")
else:
    print("The 25% improvement goal was not achieved.")