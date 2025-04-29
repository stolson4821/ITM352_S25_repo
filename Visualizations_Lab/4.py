import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the filename and get the full path to the file in the current directory
filename = "Trips_from_area_8.json"
filepath = os.path.join(os.getcwd(), filename)

# Read the JSON file (not CSV)
df = pd.read_json(filepath)

# Drop rows with missing fare or tip values
df = df.dropna(subset=['fare', 'tips'])

# Convert fare and tips columns to numeric if they aren't already
df['fare'] = pd.to_numeric(df['fare'], errors='coerce')
df['tips'] = pd.to_numeric(df['tips'], errors='coerce')

# Create scatter plot: Fare (X) vs Tip (Y)
plt.scatter(df['fare'], df['tips'], alpha=0.5, color='purple', edgecolor='k')

# Add labels and title
plt.xlabel('Fare Amount ($)')
plt.ylabel('Tip Amount ($)')
plt.title('Scatter Plot of Fare vs Tip')
plt.grid(True)
plt.tight_layout()
plt.show()

