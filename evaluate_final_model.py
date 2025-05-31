import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
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

# Best hyperparameters from the previous step
best_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 100,
    'subsample': 0.8
}

# Calculate scale_pos_weight again (as it's dependent on y_train)
purchase_counts_train = y_train.value_counts()
# Ensure both classes are present before division
if 0 in purchase_counts_train and 1 in purchase_counts_train and purchase_counts_train[1] > 0:
    scale_pos_weight_val_train = purchase_counts_train[0] / purchase_counts_train[1]
else:
    scale_pos_weight_val_train = 1 # Default if a class is missing or count is zero

print(f"Training set purchase distribution:\n{purchase_counts_train}")
print(f"Calculated scale_pos_weight for final model: {scale_pos_weight_val_train:.2f}")

# Initialize XGBClassifier with best parameters
final_xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False, # Deprecated, explicitly set to False
    scale_pos_weight=scale_pos_weight_val_train,
    random_state=42,
    **best_params
)

print("\nTraining final model with best hyperparameters...")
final_xgb_clf.fit(X_train, y_train)

print("Predicting probabilities on the test set...")
y_pred_proba = final_xgb_clf.predict_proba(X_test)[:, 1]

# Calculate AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC Score on Test Set: {auc_score:.4f}")

# Check if we need to iterate
if auc_score < 0.9:
    print("\nAUC is below 0.9. Further tuning or feature engineering might be needed.")
else:
    print("\nAUC is >= 0.9. Target met!")

# Plot ROC Curve
try:
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Test Set')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve.png')
    print("\nROC curve plot saved as roc_curve.png")
except ImportError:
    print("\nMatplotlib not found. Cannot save ROC curve plot. Please install matplotlib.")
