import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("../data sets/4. dataset_tiktok_v2 - clean.csv")

# Iterate over each row
for index, row in df.iterrows():
    # Count the number of non-empty values in the "hashtag_" columns
    count = sum(1 for col in df.columns if col.startswith('hashtag_') and pd.notnull(row[col]))

    # Save the count in the corresponding "hashtag_1" column for that row
    df.at[index, 'hashtag_1'] = count

# Save the modified DataFrame back to the CSV file
df.to_csv("dataset_tiktok_v3.csv", index=False)
