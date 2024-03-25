import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("../data sets/6. dataset_tiktok_v3 - clean.csv")

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Find and print rows with empty cells
empty_rows = df[df.isnull().any(axis=1)]

# Adjust pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("Rows with empty cells:")
print(empty_rows)

df['music_name'] = df['music_name'].fillna('no sound')

empty_rows = df[df.isnull().any(axis=1)]
print(empty_rows)

df.to_csv("../data sets/7. dataset_tiktok_v4 - final.csv", index=False)
