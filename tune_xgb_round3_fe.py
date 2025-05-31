import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import numpy as np

print("--- Starting GridSearchCV Round 3 (Post-Feature Engineering) ---")

# --- Data Loading and Initial Preprocessing ---
print("\nLoading and preprocessing data...")
df_orig = pd.read_csv('price_data.csv')
categorical_cols = ['Industry', 'Product_Type', 'Demo_Type']
df_processed = pd.get_dummies(df_orig.drop('Customer_ID', axis=1), columns=categorical_cols, drop_first=True)
X_processed = df_processed.drop('Purchase', axis=1)
y = df_processed['Purchase']

# Splitting to get y_train consistent with how X_train_fe will be constructed
# Note: X_test_processed and X_test_fe are not directly used in *this* script's grid search
_, _, y_train, _ = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)
print("Data split. y_train shape:", y_train.shape)

# --- Feature Engineering ---
print("\nPerforming feature engineering...")
df_new_features = pd.DataFrame(index=df_orig.index) # Align with original df_orig index
df_new_features['Price_per_User'] = df_orig['Price_Offered'] / (df_orig['Max_Concurrent_Users'] + 1e-6)
df_new_features['Users_per_Company_Size'] = df_orig['Max_Concurrent_Users'] / (df_orig['Company_Size'] + 1e-6)
df_new_features['Price_per_Company_Size'] = df_orig['Price_Offered'] / (df_orig['Company_Size'] + 1e-6)
df_new_features.replace([np.inf, -np.inf], np.nan, inplace=True)
df_new_features.fillna(0, inplace=True)
print("New features created and NaNs handled.")

# Create X_train_fe for GridSearchCV
# We need to construct X_train_fe in the same way it was in the feature engineering step
# This means taking the X_processed that corresponds to the full dataset, adding features, then taking the train split.
# Or, more simply, recreate X_processed with new features first, then split.

X_with_all_features = X_processed.copy() # Start with OHE features for all data
# Add new features to the *entire* X_processed (before splitting) to ensure alignment
X_with_all_features['Price_per_User'] = df_new_features.loc[X_with_all_features.index, 'Price_per_User']
X_with_all_features['Users_per_Company_Size'] = df_new_features.loc[X_with_all_features.index, 'Users_per_Company_Size']
X_with_all_features['Price_per_Company_Size'] = df_new_features.loc[X_with_all_features.index, 'Price_per_Company_Size']

# Now split this combined feature set
X_train_fe, _, _, _ = train_test_split(
    X_with_all_features, y, test_size=0.2, random_state=42, stratify=y
)
# y_train is already correctly defined from the earlier split.

print("Shape of X_train_fe for GridSearchCV:", X_train_fe.shape)
if X_train_fe.isnull().sum().sum() > 0:
    print("WARNING: NaNs found in X_train_fe before tuning!")
    print(X_train_fe.isnull().sum())


# Calculate scale_pos_weight
purchase_counts_train = y_train.value_counts()
scale_pos_weight_val = purchase_counts_train[0] / purchase_counts_train[1] if 1 in purchase_counts_train and purchase_counts_train[1] > 0 and 0 in purchase_counts_train else 1
print(f"Using scale_pos_weight: {scale_pos_weight_val:.2f}")

xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, scale_pos_weight=scale_pos_weight_val, random_state=42)

# Hyperparameter grid (Round 3 - same as Round 2)
param_grid_round3 = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3]
}
print("\nHyperparameter grid defined.")

# Setup GridSearchCV
grid_search_round3 = GridSearchCV(estimator=xgb_clf, param_grid=param_grid_round3, scoring='roc_auc', cv=3, verbose=2, n_jobs=-1)

print("Starting GridSearchCV (Round 3 - Post-Feature Engineering)...")
grid_search_round3.fit(X_train_fe, y_train)

print("\nBest hyperparameters found (Round 3):")
print(grid_search_round3.best_params_)

print("\nBest AUC score on training data (during CV - Round 3):")
print(grid_search_round3.best_score_)

print("\n--- GridSearchCV Round 3 Finished ---")
