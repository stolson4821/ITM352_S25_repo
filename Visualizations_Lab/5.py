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

# a. Scatter plot using plt.scatter
plt.figure(figsize=(6, 4))
plt.scatter(df['fare'], df['trip_miles'], color='blue', alpha=0.6)
plt.xlabel('Fare ($)')
plt.ylabel('Trip Miles')
plt.title('Trip Miles vs Fare (plt.scatter)')
plt.grid(True)
plt.tight_layout()
plt.show()

# b. Scatter plot using plt.plot with linestyle="none" and marker="."
plt.figure(figsize=(6, 4))
plt.plot(df['fare'], df['trip_miles'], linestyle='none', marker='.', color='green')
plt.xlabel('Fare ($)')
plt.ylabel('Trip Miles')
plt.title('Trip Miles vs Fare (plt.plot)')
plt.grid(True)
plt.tight_layout()
plt.show()

# c. Fancy scatter plot
plt.figure(figsize=(6, 4))
plt.plot(df['fare'], df['trip_miles'], linestyle='none', marker='v', color='cyan', alpha=0.2)
plt.xlabel('Fare ($)')
plt.ylabel('Trip Miles')
plt.title('Fancy Trip Miles vs Fare Plot')
plt.grid(True)
plt.tight_layout()
plt.show()
