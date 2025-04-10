from sodapy import Socrata
import pandas as pd

# Step 1: Use Socrata client to connect and fetch data
client = Socrata("data.cityofchicago.org", None)
results = client.get("rr23-ymwb", limit=(500))

# Step 2: Convert JSON results to a pandas DataFrame
df = pd.DataFrame.from_records(results)

# Step 3: Print the first few rows to inspect the data
print("First few rows:")
print(df.head())

# Step 4: Replace NaN values in vehicle_fuel_source and vehicle_make with blank
df['vehicle_fuel_source'] = df['vehicle_fuel_source'].fillna('')
df['vehicle_make'] = df['vehicle_make'].fillna('')

# Step 5: Print vehicle_make and vehicle_fuel_source columns if they exist
if 'vehicle_make' in df.columns and 'vehicle_fuel_source' in df.columns:
    print("\nVehicle and Fuel Source:")
    print(df[['vehicle_make', 'vehicle_fuel_source']])

    # Step 6: Count number of vehicles per fuel source
    print("\nNumber of Vehicles per Fuel Source:")
    fuel_counts = df['vehicle_fuel_source'].value_counts()
    for fuel, count in fuel_counts.items():
        print(f"{fuel}: {count}")
else:
    print("vehicle_make or vehicle_fuel_source column not found in dataset.")