import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import numpy as np # Added for scale_pos_weight calculation, though not strictly needed if y_train.value_counts() is used directly.

print("--- Starting XGBoost Model Training Script ---")

# Load the dataframe
print("\nLoading data...")
df = pd.read_csv('price_data.csv')

# Drop Customer_ID
df = df.drop('Customer_ID', axis=1)
print("Dropped Customer_ID column.")

# Categorical columns for one-hot encoding
categorical_cols = ['Industry', 'Product_Type', 'Demo_Type']
print(f"Performing one-hot encoding on: {categorical_cols}...")
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("One-hot encoding complete. DataFrame shape:", df.shape)

# Separate features (X) and target (y)
X = df.drop('Purchase', axis=1)
y = df['Purchase']
print("Features (X) and target (y) separated.")

# Split data
print("\nSplitting data into training and testing sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Best hyperparameters from Round 2 tuning
best_params = {
    'colsample_bytree': 0.8,
    'gamma': 0,
    'learning_rate': 0.05,
    'max_depth': 3,
    'min_child_weight': 3,
    'n_estimators': 200,
    'subsample': 0.8
}
print(f"\nUsing best hyperparameters: {best_params}")

# Calculate scale_pos_weight for class imbalance
purchase_counts_train = y_train.value_counts()
# Ensure both classes are present before division
if 0 in purchase_counts_train and 1 in purchase_counts_train and purchase_counts_train[1] > 0:
    scale_pos_weight_val = purchase_counts_train[0] / purchase_counts_train[1]
else:
    scale_pos_weight_val = 1 # Default if a class is missing or count is zero
print(f"Calculated scale_pos_weight for training: {scale_pos_weight_val:.2f}")

# Initialize XGBClassifier with best parameters
final_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False, # Suppress the warning
    scale_pos_weight=scale_pos_weight_val,
    random_state=42,
    **best_params
)

print("\nTraining the final model...")
final_model.fit(X_train, y_train)
print("Model training complete.")

print("\nPredicting probabilities on the test set...")
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\n--- Final Model Evaluation ---")
print(f"AUC Score on Test Set: {auc_score:.4f}")

if auc_score < 0.9:
    print("NOTE: The AUC score is below the target of 0.9.")
else:
    print("Target AUC of >= 0.9 met!")

# Save the trained model
model_filename = 'final_model.joblib' # Updated to final name
print(f"\nSaving trained model to {model_filename}...")
joblib.dump(final_model, model_filename)
print(f"Model saved successfully.")

# Generate and Save ROC Curve plot
roc_plot_filename = 'final_roc_curve.png'
print(f"Generating and saving ROC curve to {roc_plot_filename}...")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Final Model')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(roc_plot_filename)
print(f"ROC curve saved to {roc_plot_filename}.")
print("\n--- Script execution finished ---")
