import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import os

# Define the filename and get the full path to the file in the current directory
filename = "Trips_from_area_8.csv"  # Update with your actual CSV filename
filepath = os.path.join(os.getcwd(), filename)

# Load the dataset
df = pd.read_csv(filepath)

# Drop rows with missing values in the necessary columns
df = df.dropna(subset=['fare', 'trip_miles', 'dropoff_area'])

# Optional: convert dropoff_area to numeric if it's not
if df['dropoff_area'].dtype == 'object':
    df['dropoff_area'] = pd.factorize(df['dropoff_area'])[0]

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plotting
ax.scatter(df['fare'], df['trip_miles'], df['dropoff_area'], 
           c='green', marker='o', alpha=0.5)

# Axis labels
ax.set_xlabel('Fare ($)')
ax.set_ylabel('Trip Miles')
ax.set_zlabel('Dropoff Area (coded)')

# Title
ax.set_title('3D Scatter Plot: Fare vs Trip Miles vs Dropoff Area')

# Show the plot
plt.tight_layout()
plt.show()
