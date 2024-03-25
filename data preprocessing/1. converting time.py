import pandas as pd
from datetime import datetime
import pytz

# Load the CSV file into a DataFrame
df = pd.read_csv("../data sets/2. dataset_tiktok.csv")

# Function to extract day of the week and time in GMT


def extract_day_and_time(iso_time):
    dt = datetime.fromisoformat(iso_time)
    day_of_week = dt.strftime("%A").lower()
    gmt = pytz.timezone('GMT')
    dt_gmt = dt.astimezone(gmt)
    time_in_gmt = dt_gmt.strftime("%H:%M:%S")
    return day_of_week, time_in_gmt


# Apply the function to each value in the "createTimeISO" column
df[['day_posted', 'time_posted']] = df['createTimeISO'].apply(lambda x: pd.Series(extract_day_and_time(x)))

# Save the modified DataFrame to a new CSV file
# df.to_csv("dataset_tiktok_v2.csv", index=False)
