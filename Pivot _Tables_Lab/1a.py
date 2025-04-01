import pandas as pd


#sales data file from drive
url = "https://drive.google.com/file/d/1ujY0WCcePdotG2xdbLyeECFW9lCJ4t-K"

try:
    df = pd.read_csv(url, engine='pyarrow', on_bad_lines="skip")

    print(df.head())
except Exception as e:
    print(f'Issue found:{e}')