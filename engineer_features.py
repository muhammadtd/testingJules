import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np # For np.isinf and np.nan

print("--- Starting Feature Engineering Script ---")

# Load the dataframe
print("\nLoading original data...")
df_orig = pd.read_csv('price_data.csv')

# Perform one-hot encoding (on a copy to keep original numeric features for FE)
print("Creating processed DataFrame for one-hot encoding...")
df_processed = pd.get_dummies(df_orig.drop('Customer_ID', axis=1),
                              columns=['Industry', 'Product_Type', 'Demo_Type'],
                              drop_first=True)
print("Processed DataFrame created. Shape:", df_processed.shape)

# Separate features (X_processed for training) and target (y)
X_processed = df_processed.drop('Purchase', axis=1)
y = df_processed['Purchase']
print("Base processed features (X_processed) and target (y) separated.")

# Split processed data to get X_train_processed, X_test_processed, y_train, y_test
print("\nSplitting processed data into training and testing sets (80/20)...")
X_train_processed, X_test_processed, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)
print(f"X_train_processed shape: {X_train_processed.shape}, X_test_processed shape: {X_test_processed.shape}")

# Create new features using the original numerical columns from df_orig
print("\nStarting feature engineering...")
df_new_features = pd.DataFrame(index=df_orig.index) # Use original DataFrame's index

# Feature Engineering
df_new_features['Price_per_User'] = df_orig['Price_Offered'] / (df_orig['Max_Concurrent_Users'] + 1e-6)
df_new_features['Users_per_Company_Size'] = df_orig['Max_Concurrent_Users'] / (df_orig['Company_Size'] + 1e-6)
df_new_features['Price_per_Company_Size'] = df_orig['Price_Offered'] / (df_orig['Company_Size'] + 1e-6)
print("New features created.")

# Handle potential infinities or NaNs
df_new_features.replace([np.inf, -np.inf], np.nan, inplace=True)
print("\nNaNs in new features before fill:")
print(df_new_features.isnull().sum())

df_new_features.fillna(0, inplace=True)
print("\nNaNs in new features after fill (should be 0):")
print(df_new_features.isnull().sum())

# Align new features with X_train_processed and X_test_processed
print("\nAdding new features to X_train and X_test sets...")
X_train_fe = X_train_processed.copy()
X_test_fe = X_test_processed.copy()

# Add new features. Indices should align correctly.
X_train_fe['Price_per_User'] = df_new_features.loc[X_train_fe.index, 'Price_per_User']
X_train_fe['Users_per_Company_Size'] = df_new_features.loc[X_train_fe.index, 'Users_per_Company_Size']
X_train_fe['Price_per_Company_Size'] = df_new_features.loc[X_train_fe.index, 'Price_per_Company_Size']

X_test_fe['Price_per_User'] = df_new_features.loc[X_test_fe.index, 'Price_per_User']
X_test_fe['Users_per_Company_Size'] = df_new_features.loc[X_test_fe.index, 'Users_per_Company_Size']
X_test_fe['Price_per_Company_Size'] = df_new_features.loc[X_test_fe.index, 'Price_per_Company_Size']

print("\nShape of X_train with new features (X_train_fe):", X_train_fe.shape)
print("First 5 rows of X_train_fe:")
print(X_train_fe.head())

print("\nShape of X_test with new features (X_test_fe):", X_test_fe.shape)
print("First 5 rows of X_test_fe:")
print(X_test_fe.head())

# Check for NaNs in the final feature sets again
print("\nTotal NaNs in X_train_fe after adding new features:")
print(X_train_fe.isnull().sum().sum())
print("\nTotal NaNs in X_test_fe after adding new features:")
print(X_test_fe.isnull().sum().sum())

# Optional: Save feature sets to CSV
# For this subtask, saving to files is not required, printing confirmation is enough.
# X_train_fe.to_csv('X_train_engineered.csv', index=False)
# X_test_fe.to_csv('X_test_engineered.csv', index=False)
# y_train.to_csv('y_train_engineered.csv', index=False) # y does not change
# y_test.to_csv('y_test_engineered.csv', index=False)
# print("\n(Optional) Engineered feature sets would be saved if uncommented.")

print("\n--- Feature Engineering Script Finished ---")
