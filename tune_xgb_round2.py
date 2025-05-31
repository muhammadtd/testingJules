import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import numpy as np

# Load the dataframe
df = pd.read_csv('price_data.csv')

# Drop Customer_ID
df = df.drop('Customer_ID', axis=1)

# Categorical columns for one-hot encoding
categorical_cols = ['Industry', 'Product_Type', 'Demo_Type']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features (X) and target (y)
X = df.drop('Purchase', axis=1)
y = df['Purchase']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate scale_pos_weight
purchase_counts_train = y_train.value_counts()
if 1 in purchase_counts_train and purchase_counts_train[1] > 0 and 0 in purchase_counts_train:
    scale_pos_weight_val = purchase_counts_train[0] / purchase_counts_train[1]
else:
    scale_pos_weight_val = 1 # Default if a class is missing or count is zero

print(f"Using scale_pos_weight: {scale_pos_weight_val:.2f}")

xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, scale_pos_weight=scale_pos_weight_val, random_state=42)

# Define the expanded hyperparameter grid for Round 2
param_grid_round2 = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3]
}

# Setup GridSearchCV
grid_search_round2 = GridSearchCV(estimator=xgb_clf, param_grid=param_grid_round2, scoring='roc_auc', cv=3, verbose=2, n_jobs=-1)

print("Starting GridSearchCV (Round 2)...")
grid_search_round2.fit(X_train, y_train)

print("\nBest hyperparameters found (Round 2):")
print(grid_search_round2.best_params_)

print("\nBest AUC score on training data (during CV - Round 2):")
print(grid_search_round2.best_score_)
