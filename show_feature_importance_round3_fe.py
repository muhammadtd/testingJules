import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier # Required for model loading if custom class
import numpy as np # For X_test_fe column recreation

# --- Recreate feature names for X_train_fe / X_test_fe ---
# This is crucial for correctly labeling the feature importances.
# We need the exact column order as it was when the model was trained.

# Load original data to get initial column names after one-hot encoding
df_orig = pd.read_csv('price_data.csv')
df_orig = df_orig.drop('Customer_ID', axis=1)
categorical_cols = ['Industry', 'Product_Type', 'Demo_Type']
df_orig_encoded = pd.get_dummies(df_orig, columns=categorical_cols, drop_first=True)
original_features_after_encoding = df_orig_encoded.drop('Purchase', axis=1).columns.tolist()

# Define the engineered feature names
engineered_feature_names = ['Price_per_User', 'Users_per_Company_Size', 'Price_per_Company_Size']

# Combine them to get the full list of feature names for the FE model
# The engineered features were added to the end of the one-hot encoded features.
feature_names_fe = original_features_after_encoding + engineered_feature_names
print(f"Reconstructed feature names for the model: {feature_names_fe}")
print(f"Number of features: {len(feature_names_fe)}")

# Load the trained model
model_filename = 'xgboost_model_round3_fe.joblib'
print(f"Loading model: {model_filename}")
loaded_model = joblib.load(model_filename)

# Get feature importances
importances = loaded_model.feature_importances_
print(f"Number of importances received: {len(importances)}")

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'feature': feature_names_fe,
    'importance': importances
})

# Sort by importance
importance_df = importance_df.sort_values(by='importance', ascending=False)

print("\nFeature Importances (Round 3 FE Model):")
print(importance_df)

# Plot feature importances
plt.figure(figsize=(10, 8))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel("XGBoost Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - Round 3 (Feature Engineered) Model")
plt.gca().invert_yaxis() # Display features with highest importance at the top
plt.tight_layout()
plot_filename = 'feature_importance_round3_fe.png'
plt.savefig(plot_filename)
print(f"\nFeature importance plot saved to {plot_filename}")

print("\nScript show_feature_importance_round3_fe.py finished.")
