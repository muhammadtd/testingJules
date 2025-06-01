import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

print("--- Starting Random Forest Model Evaluation Script ---")

# --- Data Loading and Initial Preprocessing (Replicating) ---
print("\nLoading and preprocessing data...")
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
print("X_test_processed shape:", X_test_processed.shape)


# --- Feature Engineering (Replicating) ---
print("\nPerforming feature engineering for train and test sets...")
# Create a temporary DataFrame for new features based on the original df_orig's index
df_temp_features_for_alignment = df_orig[['Price_Offered', 'Max_Concurrent_Users', 'Company_Size']].copy()

df_temp_features_for_alignment['Price_per_User'] = df_temp_features_for_alignment['Price_Offered'] / (df_temp_features_for_alignment['Max_Concurrent_Users'] + 1e-6)
df_temp_features_for_alignment['Users_per_Company_Size'] = df_temp_features_for_alignment['Max_Concurrent_Users'] / (df_temp_features_for_alignment['Company_Size'] + 1e-6)
df_temp_features_for_alignment['Price_per_Company_Size'] = df_temp_features_for_alignment['Price_Offered'] / (df_temp_features_for_alignment['Company_Size'] + 1e-6)

df_temp_features_for_alignment.replace([np.inf, -np.inf], np.nan, inplace=True)
df_temp_features_for_alignment.fillna(0, inplace=True)

# Align new features with the train/test sets using their original indices
X_train_fe = X_train_processed.copy()
X_test_fe = X_test_processed.copy()

X_train_fe['Price_per_User'] = df_temp_features_for_alignment.loc[X_train_fe.index, 'Price_per_User']
X_train_fe['Users_per_Company_Size'] = df_temp_features_for_alignment.loc[X_train_fe.index, 'Users_per_Company_Size']
X_train_fe['Price_per_Company_Size'] = df_temp_features_for_alignment.loc[X_train_fe.index, 'Price_per_Company_Size']

X_test_fe['Price_per_User'] = df_temp_features_for_alignment.loc[X_test_fe.index, 'Price_per_User']
X_test_fe['Users_per_Company_Size'] = df_temp_features_for_alignment.loc[X_test_fe.index, 'Users_per_Company_Size']
X_test_fe['Price_per_Company_Size'] = df_temp_features_for_alignment.loc[X_test_fe.index, 'Price_per_Company_Size']

# Robustness: Fill any NaNs that might have been created if indices didn't perfectly align or from original data
if X_train_fe.isnull().sum().any():
    print("Warning: NaNs found in X_train_fe after feature engineering. Filling with 0.")
    X_train_fe.fillna(0, inplace=True)
if X_test_fe.isnull().sum().any():
    print("Warning: NaNs found in X_test_fe after feature engineering. Filling with 0.")
    X_test_fe.fillna(0, inplace=True)

print("Feature engineering complete.")
print("Shape of X_train_fe for RF training:", X_train_fe.shape)
print("Shape of X_test_fe for RF evaluation:", X_test_fe.shape)
# --- End of Replicated Data Prep ---


# Best hyperparameters for RandomForest from the previous step
best_rf_params = {
    'max_depth': 5,
    'min_samples_leaf': 4,
    'min_samples_split': 2,
    'n_estimators': 200,
    'class_weight': 'balanced_subsample', # From RF initialization in previous script
    'random_state': 42 # For reproducibility
}
print(f"\nUsing best Random Forest hyperparameters: {best_rf_params}")

# Initialize RandomForestClassifier with best parameters
final_rf_clf = RandomForestClassifier(**best_rf_params)

print("\nTraining final Random Forest model with best hyperparameters...")
final_rf_clf.fit(X_train_fe, y_train)
print("Random Forest Model training complete.")

print("\nPredicting probabilities on the feature-engineered test set...")
y_pred_proba_rf = final_rf_clf.predict_proba(X_test_fe)[:, 1]

# Calculate AUC for Random Forest
auc_score_rf = roc_auc_score(y_test, y_pred_proba_rf)
print(f"\n--- Random Forest Model Evaluation ---")
print(f"AUC Score on Test Set (Random Forest with FE): {auc_score_rf:.4f}")

if auc_score_rf < 0.9:
    print("AUC for Random Forest is below 0.9.")
else:
    print("AUC for Random Forest is >= 0.9. Target met!")

# Plot ROC Curve for Random Forest model
roc_plot_filename_rf = 'roc_curve_rf.png'
print(f"\nGenerating and saving ROC curve to {roc_plot_filename_rf}...")
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest ROC curve (AUC = {auc_score_rf:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Random Forest (Feature Engineered)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(roc_plot_filename_rf)
print(f"ROC curve plot for Random Forest saved as {roc_plot_filename_rf}.")

print("\n--- Script evaluate_rf_model.py finished ---")
