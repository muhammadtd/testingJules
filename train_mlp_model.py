import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# 1. Load preprocessed data
data = joblib.load('processed_data.joblib')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# Convert boolean columns to int (if any) that might have been loaded as object type by pandas in X_train
# This is a precaution, as MLPClassifier expects numeric input.
# One-hot encoding in preprocess_data.py should produce boolean True/False which are fine.
# However, if they were loaded as strings 'True'/'False', this would be an issue.
# Let's ensure all columns are numeric.
if isinstance(X_train, pd.DataFrame):
    for col in X_train.columns:
        if X_train[col].dtype == 'bool':
            X_train[col] = X_train[col].astype(int)
            X_test[col] = X_test[col].astype(int)
elif hasattr(X_train, 'columns'): # Handle cases where it might be a non-DataFrame but has columns
    # This part might need adjustment based on actual data structure if not pandas DataFrame
    # For now, assuming pandas DataFrame as per preprocess_data.py
    pass


print("Data loaded successfully.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# 2. Imports are already at the top

# 3. Define a parameter grid for hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive']
}

# 4. Initialize MLPClassifier
mlp = MLPClassifier(random_state=42, max_iter=1000) # Increased max_iter for convergence

# 5. Initialize GridSearchCV
# Using n_jobs=-1 to use all available cores, making it faster if possible.
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

# 6. Fit GridSearchCV on X_train and y_train
print("\nStarting GridSearchCV...")
grid_search.fit(X_train, y_train)
print("GridSearchCV fitting completed.")

# 7. Print the best parameters and the best AUC score
print("\nBest Parameters found by GridSearchCV:")
print(grid_search.best_params_)
print("Best ROC AUC score from GridSearchCV:")
print(grid_search.best_score_)

# 8. Train a final MLPClassifier model using the best parameters
print("\nTraining final model with best parameters...")
best_mlp = MLPClassifier(**grid_search.best_params_, random_state=42, max_iter=1000) # Increased max_iter
best_mlp.fit(X_train, y_train)
print("Final model training completed.")

# 9. Save the trained model
model_filename = 'mlp_purchase_model.joblib'
joblib.dump(best_mlp, model_filename)
print(f"\nTrained model saved as {model_filename}")

print("\nScript finished.")
