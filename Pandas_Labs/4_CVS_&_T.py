import requests
import pandas as pd

# Direct link to the file, replace 'FILE_ID' with the actual file ID from the Google Drive link
file_id = '1-MpDUIRZxhFnN-rcDdJQMe_mcCSciaus'
url = f'https://drive.google.com/uc?id={file_id}'

# Download the file
response = requests.get(url)
file_name = 'taxi_trip_data.json'

# Save the content to a file
with open(file_name, 'wb') as file:
    file.write(response.content)

# Reading the JSON file into a DataFrame
df = pd.read_json(file_name)

print(f"Fare Median is:", df['fare'].median())