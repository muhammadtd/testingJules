import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve
import matplotlib.pyplot as plt

# 1. Imports are updated

# 2. Load the trained MLP model
model_filename = 'mlp_purchase_model.joblib'
try:
    model = joblib.load(model_filename)
    print(f"Model '{model_filename}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found. Please run the training script first.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 3. Load the preprocessed test data
data_filename = 'processed_data.joblib'
try:
    data = joblib.load(data_filename)
    X_test = data['X_test']
    y_test = data['y_test']
    print(f"Data '{data_filename}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Data file '{data_filename}' not found. Please run the preprocessing script first.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Ensure X_test is in the correct format (e.g., converting boolean columns if necessary)
# This step should align with how X_train was prepared before fitting the model in train_mlp_model.py
if isinstance(X_test, pd.DataFrame):
    for col in X_test.columns:
        if X_test[col].dtype == 'bool':
            X_test[col] = X_test[col].astype(int)
elif hasattr(X_test, 'columns'):
    # This part might need adjustment based on actual data structure if not pandas DataFrame
    pass

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# 4. Make predictions
# Predict probabilities for AUC score (typically for the positive class)
try:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
except Exception as e:
    print(f"Error during probability prediction: {e}")
    # Potentially print X_test.dtypes or other debug info here
    # For example:
    # if isinstance(X_test, pd.DataFrame):
    #     print("X_test dtypes:\n", X_test.dtypes)
    # else:
    #     print("X_test type:", type(X_test))
    #     if hasattr(X_test, 'dtype'):
    #         print("X_test dtype:", X_test.dtype)
    exit()


# Predict class labels for accuracy, precision, recall
try:
    y_pred_class = model.predict(X_test)
except Exception as e:
    print(f"Error during class prediction: {e}")
    exit()

print("\nPredictions made successfully.")

# 5. Calculate AUC score
try:
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC Score: {auc_score:.4f}")
except Exception as e:
    print(f"Error calculating AUC score: {e}")
    auc_score = None # Ensure variable exists

# 6. Calculate accuracy, precision, and recall
try:
    accuracy = accuracy_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class, zero_division=0) # Handles cases with no positive predictions
    recall = recall_score(y_test, y_pred_class, zero_division=0) # Handles cases with no actual positives

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
except Exception as e:
    print(f"Error calculating other metrics: {e}")

# 2. Calculate ROC curve
try:
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
except Exception as e:
    print(f"Error calculating ROC curve values: {e}")
    fpr, tpr = None, None # Ensure variables exist

# 3. Plot ROC curve
if fpr is not None and tpr is not None and auc_score is not None:
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Random classifier
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for MLP Model')
        plt.legend(loc="lower right")

        # 4. Save the plot
        plot_filename = 'mlp_roc_curve.png'
        plt.savefig(plot_filename)
        print(f"\nROC curve plot saved as {plot_filename}")
        # plt.show() # Typically not used in scripts, but useful for interactive sessions
    except Exception as e:
        print(f"Error generating or saving ROC plot: {e}")
else:
    print("\nSkipping ROC curve plotting due to earlier errors.")

print("\nScript finished.")
