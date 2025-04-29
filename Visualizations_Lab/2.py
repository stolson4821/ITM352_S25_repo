import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the filename and get the full path to the file in the current directory
filename = "Trips_from_area_8.json"
filepath = os.path.join(os.getcwd(), filename)

# Load the JSON file
df = pd.read_json(filepath)

# a. Create a histogram using the 'trip_miles' column
plt.hist(df['trip_miles'], bins= 15, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Trip Miles')
plt.ylabel('Frequency')
plt.title('Histogram of Trip Miles')

# Show plot
plt.grid(True)
plt.show()

