import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the filename and get the full path to the file in the current directory
filename = "Trips_from_area_8.csv"  # Update with your actual CSV filename
filepath = os.path.join(os.getcwd(), filename)

# Read the CSV file
df = pd.read_csv(filepath)

# Drop rows with missing values in 'fare' or 'trip_miles'
df = df.dropna(subset=['fare', 'trip_miles'])

# b. Filter out trips of 0 miles
df = df[df['trip_miles'] != 0]

# c. Filter out trips less than 2 miles
df = df[df['trip_miles'] >= 2]

# Create scatter plot: Fare vs Trip Miles
plt.figure(figsize=(8, 6))
plt.scatter(df['fare'], df['trip_miles'], color='blue', alpha=0.5)
plt.xlabel('Fare ($)')
plt.ylabel('Trip Miles')
plt.title('Fare vs Trip Miles (Trips ≥ 2 miles)')
plt.grid(True)
plt.tight_layout()

# a. Save the plot to a file
plt.savefig("FaresXmiles.png")

# Show plot (optional if just saving)
plt.show()
