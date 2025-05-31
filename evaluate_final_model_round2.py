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

# Best hyperparameters from Round 2
best_params_round2 = {
    'colsample_bytree': 0.8,
    'gamma': 0,
    'learning_rate': 0.05,
    'max_depth': 3,
    'min_child_weight': 3,
    'n_estimators': 200,
    'subsample': 0.8
}

# Calculate scale_pos_weight
purchase_counts_train = y_train.value_counts()
if 1 in purchase_counts_train and purchase_counts_train[1] > 0 and 0 in purchase_counts_train:
    scale_pos_weight_val_train = purchase_counts_train[0] / purchase_counts_train[1]
else:
    scale_pos_weight_val_train = 1 # Default

print(f"Using scale_pos_weight: {scale_pos_weight_val_train:.2f}")

# Initialize XGBClassifier with best parameters from Round 2
final_xgb_clf_round2 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight_val_train,
    random_state=42,
    **best_params_round2
)

print("Training final model with best hyperparameters (Round 2)...")
final_xgb_clf_round2.fit(X_train, y_train)

print("Predicting probabilities on the test set (Round 2)...")
y_pred_proba_round2 = final_xgb_clf_round2.predict_proba(X_test)[:, 1]

# Calculate AUC for Round 2
auc_score_round2 = roc_auc_score(y_test, y_pred_proba_round2)
print(f"\nAUC Score on Test Set (Round 2): {auc_score_round2:.4f}")

if auc_score_round2 < 0.9:
    print("\nAUC (Round 2) is below 0.9.")
else:
    print("\nAUC (Round 2) is >= 0.9. Target met!")

# Plot ROC Curve for Round 2 model
try:
    fpr_round2, tpr_round2, _ = roc_curve(y_test, y_pred_proba_round2)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_round2, tpr_round2, color='blue', lw=2, label=f'ROC curve (Round 2 AUC = {auc_score_round2:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Test Set (Round 2)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve_round2.png')
    print("\nROC curve plot (Round 2) saved as roc_curve_round2.png")
except ImportError:
    print("\nMatplotlib not found. Cannot save ROC curve plot.")
except Exception as e:
    print(f"\nAn error occurred during ROC curve plotting: {e}")
