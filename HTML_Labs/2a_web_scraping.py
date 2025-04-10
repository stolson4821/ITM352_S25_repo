import pandas as pd
import ssl

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# URL containing the Treasury yield curve data
url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month=202410"

# Extract all tables from the URL
tables = pd.read_html(url)

# Print number of tables found
print(f"Number of tables found: {len(tables)}")

# Display columns of the first table (typically the interest rate data)
df = tables[0]
print("Columns of the interest rate table:")
print(df.columns)