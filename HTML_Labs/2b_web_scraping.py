import pandas as pd
import ssl

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# URL of the Treasury yield data
url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month=202410"

# Read tables from the URL
tables = pd.read_html(url)

# Print number of tables found
print(f"Number of tables found: {len(tables)}")

df = tables[0]

# Print header for clarity
print("Date       | 1 Mo Rate")
print("------------------------")

# Loop through the rows using iterrows and print 1-month interest rates
for index, row in df.iterrows():
    date = row['Date']
    rate_1mo = row['1 Mo'] if '1 Mo' in row else 'N/A'
    print(f"{date} | {rate_1mo}")
