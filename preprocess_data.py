import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataframe
df = pd.read_csv('price_data.csv')

# Drop Customer_ID
df = df.drop('Customer_ID', axis=1)

# Identify categorical columns
categorical_cols = ['Industry', 'Product_Type', 'Demo_Type']

# Perform one-hot encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features (X) and target (y)
X = df.drop('Purchase', axis=1)
y = df['Purchase']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
numerical_cols = ['Company_Size', 'Past_Purchases']
scaler = StandardScaler()

X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

import joblib

# Print shapes
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
print("\nFirst 5 rows of X_train:")
print(X_train.head())

# Save the processed data
data_to_save = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test
}
joblib.dump(data_to_save, 'processed_data.joblib')
print("\nProcessed data saved to processed_data.joblib")
