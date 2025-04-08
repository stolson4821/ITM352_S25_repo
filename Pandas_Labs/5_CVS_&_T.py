import pandas as pd
import requests
from io import StringIO

# Google Drive file ID
file_id = "1M-X_bypJJ6K5p6eM6aYBwt1qIizIiIex"

# Correct direct download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Fetch content with requests (handling SSL issues)
response = requests.get(url, verify=False)  # Disable SSL verification (temporary fix)

# Check if request was successful
if response.status_code == 200:
    csv_data = StringIO(response.text)  # Convert response content to file-like object
    df = pd.read_csv(csv_data)  # Read CSV data into DataFrame
    #print(df.head(10))
else:
    print("Failed to download file. Check the file ID and permissions.")

###For the following only activate the letter code at a time. Each one is made so that 
###they can be activated one at a time or ran all simultaniously.
#A. Print out the dimensions of the data frame and show the first 10 rows:
#print(df)
    #print(f'The dimentions are {df.shape}')
    #print(df.head())

#B. Select only properties that have 500 or more units. 
#Drop some unnecessary columns and print the first 10 rows
#df['units'] = pd.to_numeric(df['units'], errors='coerce')

#df_filtered = df[df['units'] >= 500]
#df_filtered = df_filtered.drop(columns=['id', 'borough', 'easement'])

    #print(df_filtered.head())

#C. Look at the data types. Determine which data types are incorrect and 
# coerce them to the correct data type. Look at the data types now and print 
# the cleaned data
#df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
#df['land_sqft'] = pd.to_numeric(df['land_sqft'], errors='coerce')
#df['gross_sqft'] = pd.to_numeric(df['gross_sqft'], errors='coerce')
#print(df_filtered.head())

#D We have some null values and duplicates.  Drop those rows and print out the results. 
#df_filtered = df_filtered.drop(columns=['id', 'borough', 'easement'])
#df filtered = df_filtered.drop(columns=)
#print()

#E Filter out 0 sales and print the results. Compute and display the average sales price 
#print()

#F What is the purpose of using a CSV file rather than JSON or another data format?
#print()