import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Required for heatmap

# Load the CSV from Google Drive
url = "https://drive.google.com/uc?id=1-_yuyIroypBZpe20zGDNT9tU33emEN2q"
df = pd.read_csv(url)

# Drop missing values in the key columns
df = df.dropna(subset=['pickup_community_area', 'dropoff_community_area'])

# Create a crosstab of counts between pickup and dropoff areas
heatmap_data = pd.crosstab(df['pickup_community_area'], df['dropoff_community_area'])

# Set up the plot
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=0.5, annot=False)

# Add labels and title
plt.title('Heatmap of Pickup vs Dropoff Community Areas')
plt.xlabel('Dropoff Community Area')
plt.ylabel('Pickup Community Area')
plt.tight_layout()

# Show the plot
plt.show()
