import pandas as pd

try:
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv('price_data.csv')

    # Print the first 5 rows
    print("First 5 rows:")
    print(df.head())

    # Print the DataFrame's info
    print("\nDataFrame info:")
    df.info()

    # Print descriptive statistics
    print("\nDescriptive statistics:")
    print(df.describe(include='all'))

except ImportError:
    print("pandas library is not installed. Please install it to run this script.")
except FileNotFoundError:
    print("Error: price_data.csv not found. Make sure the file is in the correct directory.")
