import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import numpy as np # Will be used for scale_pos_weight calculation

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

# Initialize XGBClassifier
# Use scale_pos_weight for imbalanced classes
purchase_counts = y_train.value_counts()
print(f"Purchase distribution in training set:\n{purchase_counts}")
if purchase_counts[1] > 0 and purchase_counts[0] > 0:
    scale_pos_weight_val = purchase_counts[0] / purchase_counts[1]
else:
    scale_pos_weight_val = 1 # Default if one class is missing (should not happen with stratify)
print(f"Calculated scale_pos_weight: {scale_pos_weight_val:.2f}")

xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, scale_pos_weight=scale_pos_weight_val, random_state=42)

# Define a comprehensive hyperparameter grid
param_grid = {
    'n_estimators': [100, 200], # Reduced for faster initial test
    'max_depth': [3, 5],        # Reduced for faster initial test
    'learning_rate': [0.01, 0.1], # Reduced for faster initial test
    'subsample': [0.8, 1.0],      # Reduced for faster initial test
    'colsample_bytree': [0.8, 1.0] # Reduced for faster initial test
}

# Setup GridSearchCV
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=2, n_jobs=-1)

print("Starting GridSearchCV...")
grid_search.fit(X_train, y_train)

print("\nBest hyperparameters found:")
print(grid_search.best_params_)

print("\nBest AUC score on training data (during CV):")
print(grid_search.best_score_)
