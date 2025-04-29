import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the filename and get the full path to the file in the current directory
filename = "Trips_from_area_8.json"
filepath = os.path.join(os.getcwd(), filename)

# Load the JSON file
df = pd.read_json(filepath)

# Check for columns and structure in the JSON file (optional)
# print(df.head())  # Uncomment this line if you need to check the structure

# b. Drop rows with NA values in the 'payment_type' or 'tips' columns
df = df.dropna(subset=['payment_type', 'tips'])

# a. Group by payment method and sum tips
tips_by_payment = df.groupby('payment_type')['tips'].sum().sort_values()

# Plot as a bar chart
tips_by_payment.plot(kind='bar', color='orange', edgecolor='black')


# c. Add labels and title
plt.xlabel('Payment Method')
plt.ylabel('Total Tips ($)')
plt.title('Total Tips by Payment Method')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Show plot
plt.tight_layout()
plt.show()
