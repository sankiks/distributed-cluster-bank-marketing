import pandas as pd

# 1. Load the dataset (note the separator is semicolon ';')
file_path = 'bank-full.csv'
df = pd.read_csv(file_path, sep=';')

# 2. Display the first 5 rows to inspect the structure and columns
print("--- Sample Data (First 5 Rows) ---")
print(df.head())

# 3. Check data types, non-null counts, and shape of the dataset
print("\n--- Dataset Info ---")
print(df.info())

# 4. Check for missing (null) values in each column
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# 5. Show basic statistical summary of numerical columns (like balance, age)
print("\n--- Statistical Summary ---")
print(df.describe())