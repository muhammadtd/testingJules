import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- Data Loading and Initial Preprocessing (Replicating from previous steps) ---
print("--- Starting RandomForest Hyperparameter Tuning Script ---")
print("Loading and preprocessing data...")
df_orig = pd.read_csv('price_data.csv')
categorical_cols = ['Industry', 'Product_Type', 'Demo_Type']
df_processed = pd.get_dummies(df_orig.drop('Customer_ID', axis=1), columns=categorical_cols, drop_first=True)
X_processed = df_processed.drop('Purchase', axis=1)
y = df_processed['Purchase']
X_train_processed, X_test_processed, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)
print("Initial data preprocessing complete.")
print("X_train_processed shape:", X_train_processed.shape)

# --- Feature Engineering (Replicating from Step 7 / previous context) ---
print("\nPerforming feature engineering...")
# Create a temporary DataFrame for new features based on the original df_orig's index
# This ensures correct alignment before merging/assigning to X_train_fe
df_temp_features_for_alignment = df_orig[['Price_Offered', 'Max_Concurrent_Users', 'Company_Size']].copy()

df_temp_features_for_alignment['Price_per_User'] = df_temp_features_for_alignment['Price_Offered'] / (df_temp_features_for_alignment['Max_Concurrent_Users'] + 1e-6)
df_temp_features_for_alignment['Users_per_Company_Size'] = df_temp_features_for_alignment['Max_Concurrent_Users'] / (df_temp_features_for_alignment['Company_Size'] + 1e-6)
df_temp_features_for_alignment['Price_per_Company_Size'] = df_temp_features_for_alignment['Price_Offered'] / (df_temp_features_for_alignment['Company_Size'] + 1e-6)

df_temp_features_for_alignment.replace([np.inf, -np.inf], np.nan, inplace=True)
df_temp_features_for_alignment.fillna(0, inplace=True) # Fill NaNs that might arise from division by zero if 1e-6 was not used or if original data had NaNs

# Now, carefully align these engineered features with X_train_fe and X_test_fe
X_train_fe = X_train_processed.copy()
# X_test_fe would be prepared similarly if evaluation was part of this script

# Assign engineered features using the index of X_train_fe to pull from df_temp_features_for_alignment
X_train_fe['Price_per_User'] = df_temp_features_for_alignment.loc[X_train_fe.index, 'Price_per_User']
X_train_fe['Users_per_Company_Size'] = df_temp_features_for_alignment.loc[X_train_fe.index, 'Users_per_Company_Size']
X_train_fe['Price_per_Company_Size'] = df_temp_features_for_alignment.loc[X_train_fe.index, 'Price_per_Company_Size']

print("Feature engineering complete.")
print("Shape of X_train_fe for RandomForest GridSearchCV:", X_train_fe.shape)
# Check for NaNs after feature engineering
if X_train_fe.isnull().sum().any():
    print("Warning: NaNs found in X_train_fe after feature engineering. Filling with 0.")
    X_train_fe.fillna(0, inplace=True) # Robustness check
# --- End of Replicated Data Prep ---

# Initialize RandomForestClassifier
# Using class_weight='balanced' or 'balanced_subsample' for imbalanced classes.
rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced_subsample')
print("\nInitialized RandomForestClassifier with class_weight='balanced_subsample'.")

# Define a hyperparameter grid for RandomForest
# Reduced grid for faster execution. Can be expanded.
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None], # None means nodes are expanded until all leaves are pure
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
    # 'max_features': ['auto', 'sqrt', 'log2'] # 'auto' is often same as 'sqrt' for RF
}
print(f"Hyperparameter grid for RandomForest: {param_grid_rf}")

# Setup GridSearchCV
# Using cv=3 for faster execution.
grid_search_rf = GridSearchCV(estimator=rf_clf, param_grid=param_grid_rf, scoring='roc_auc', cv=3, verbose=2, n_jobs=-1)

print("\nStarting GridSearchCV for RandomForestClassifier...")
grid_search_rf.fit(X_train_fe, y_train) # Using feature-engineered training data

print("\n--- Results ---")
print("Best hyperparameters found for RandomForestClassifier:")
print(grid_search_rf.best_params_)

print("\nBest AUC score on training data (during CV - RandomForest):")
print(f"{grid_search_rf.best_score_:.4f}")

print("\n--- Script full_rf_tuning_script.py finished ---")
