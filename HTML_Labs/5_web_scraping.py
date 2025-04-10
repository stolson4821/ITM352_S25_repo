import requests
import pandas as pd

# Step 1: Make a GET request to the URL
url = "https://data.cityofchicago.org/resource/97wa-y6ff.json?$select=driver_type,count(license)&$group=driver_type"
response = requests.get(url)

# Step 2: Convert the response to JSON format
data = response.json()

# Step 3: Print the response to inspect the structure
print("Response data:")
print(data)
print("\nWhat format is this data?")

# Step 4: Convert the JSON data to a pandas DataFrame
df = pd.DataFrame(data)

# Step 5: Set the column names to "count" and "driver_type"
df.columns = ['driver_type', 'count']

# Step 6: Set the index to 'driver_type' column
df.set_index('driver_type', inplace=True)

# Step 7: Convert the 'count' column to integers
df['count'] = pd.to_numeric(df['count'], errors='coerce')

# Step 8: Print the DataFrame to display the number of licenses by driver type
print("\nNumber of Licenses by Driver Type:")
print(df)