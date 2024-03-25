import pandas as pd

# # Load the CSV file into a DataFrame
# df = pd.read_csv("../data sets/6. dataset_tiktok_v3 - clean.csv")
#
# # Drop duplicate rows
# df.drop_duplicates(inplace=True)
#
# # Find and print rows with empty cells
# empty_rows = df[df.isnull().any(axis=1)]
#
# # Adjust pandas display options to show all rows and columns
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
#
# print("Rows with empty cells:")
# print(empty_rows)
#
# df['music_name'] = df['music_name'].fillna('no sound')
#
# empty_rows = df[df.isnull().any(axis=1)]
# print(empty_rows)
#
# df.to_csv("../data sets/7. dataset_tiktok_v4 - final.csv", index=False)

# Read the CSV file
df = pd.read_csv("../data sets/7. dataset_tiktok_v4 - final.csv")

# Loop through each row and convert time to decimal
for index, row in df.iterrows():
    # Split the time string by ':' and take the first two elements
    time_parts = row['time_posted'].split(':')[:2]
    # Join the first two parts with a '.' to form the new time string
    new_time = '.'.join(time_parts)
    # Update the time_posted column with the new time string
    df.at[index, 'time_posted'] = new_time

# Save the updated DataFrame back to the original CSV file
df.to_csv("../data sets/7. dataset_tiktok_v4 - final.csv", index=False)

