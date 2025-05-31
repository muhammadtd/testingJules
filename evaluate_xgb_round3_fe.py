import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib

# Load data
df = pd.read_csv('price_data.csv')

# Preprocessing
df = df.drop('Customer_ID', axis=1)
categorical_cols = ['Industry', 'Product_Type', 'Demo_Type']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features (X) and target (y)
y = df['Purchase']
X = df.drop('Purchase', axis=1)

# Split data (original features for now, will be transformed)
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Feature Engineering ---
# Replicate feature engineering from engineer_features.py for robustness
# Ensure X_train_fe and X_test_fe are created correctly based on the original splits

# Create copies to avoid SettingWithCopyWarning
X_train_fe = X_train_orig.copy()
X_test_fe = X_test_orig.copy()

# Price_per_User
X_train_fe['Price_per_User'] = X_train_fe['Price_Offered'] / (X_train_fe['Max_Concurrent_Users'] + 1e-6)
X_test_fe['Price_per_User'] = X_test_fe['Price_Offered'] / (X_test_fe['Max_Concurrent_Users'] + 1e-6)

# Users_per_Company_Size
X_train_fe['Users_per_Company_Size'] = X_train_fe['Max_Concurrent_Users'] / (X_train_fe['Company_Size'] + 1e-6)
X_test_fe['Users_per_Company_Size'] = X_test_fe['Max_Concurrent_Users'] / (X_test_fe['Company_Size'] + 1e-6)

# Price_per_Company_Size
X_train_fe['Price_per_Company_Size'] = X_train_fe['Price_Offered'] / (X_train_fe['Company_Size'] + 1e-6)
X_test_fe['Price_per_Company_Size'] = X_test_fe['Price_Offered'] / (X_test_fe['Company_Size'] + 1e-6)

# Fill potential NaNs/Infs (though 1e-6 should prevent division by zero)
X_train_fe.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_fe.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train_fe.fillna(0, inplace=True) # Using 0 as a neutral fill, could be median/mean
X_test_fe.fillna(0, inplace=True)

print("Shape of X_train_fe after engineering:", X_train_fe.shape)
print("Shape of X_test_fe after engineering:", X_test_fe.shape)
print("NaNs in X_train_fe after engineering:", X_train_fe.isnull().sum().sum())
print("NaNs in X_test_fe after engineering:", X_test_fe.isnull().sum().sum())


# Calculate scale_pos_weight for class imbalance
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}") # Should be around 4.24

# Best hyperparameters from Round 3 (feature engineered data)
best_params_round3_fe = {
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'learning_rate': 0.05,
    'max_depth': 3,
    'min_child_weight': 1,
    'n_estimators': 100,
    'subsample': 0.9
}

# Initialize and train the XGBoost model
model_round3_fe = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    **best_params_round3_fe
)

print("\nTraining model with best params from Round 3 on feature-engineered data...")
model_round3_fe.fit(X_train_fe, y_train)

# Make predictions on the feature-engineered test set
y_pred_proba_round3_fe = model_round3_fe.predict_proba(X_test_fe)[:, 1]

# Calculate ROC AUC score
roc_auc_round3_fe = roc_auc_score(y_test, y_pred_proba_round3_fe)
print(f"ROC AUC Score on Test Set (Round 3 FE Tuned): {roc_auc_round3_fe:.4f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_round3_fe)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'XGBoost (AUC = {roc_auc_round3_fe:.4f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Round 3 (Feature Engineered) Tuned Model')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve_round3_fe_tuned.png')
print("ROC curve saved to roc_curve_round3_fe_tuned.png")

# Compare with previous best test score (0.8604 from Round 2)
previous_best_auc = 0.8604 # From evaluate_xgb_round2.py
print(f"\nPrevious best test AUC (Round 2): {previous_best_auc:.4f}")
if roc_auc_round3_fe > previous_best_auc:
    print("Improvement achieved with feature engineering and Round 3 tuning!")
else:
    print("No improvement over Round 2 test AUC.")

# Save the trained model
model_filename = 'xgboost_model_round3_fe.joblib'
joblib.dump(model_round3_fe, model_filename)
print(f"Model from Round 3 (feature engineered) saved as {model_filename}")

print("\nScript evaluate_xgb_round3_fe.py finished.")
