import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Divided the temporary data equally into validation and testing
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
print(X_train["customer_zip_code_prefix"].nunique())

# Check the number of customer and seller states
print("\nCustomer states:")
print(X_train["customer_state"].nunique())

print("\nSeller states:")
print(X_train["seller_state"].nunique())





