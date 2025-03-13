import csv

# Add File name
filename = "taxi_1000.csv"

# Variables to store for calculations
total_fare = 0
fare_count = 0
max_trip_miles = 0

# Read CSV file
with open(filename, mode="r", newline="") as file:
    reader = csv.DictReader(file)  # Read CSV as a dictionary
    for row in reader:
        try:
            # Get fare value and float it
            fare = float(row["Fare"])  
            total_fare += fare
            fare_count += 1

            # Get trip miles and float it
            trip_miles = float(row["Trip Miles"])
            if trip_miles > max_trip_miles:
                max_trip_miles = trip_miles

        except ValueError:
            print(f"Skipping invalid row: {row}")  # Handle errors with skip message

# Calculate the average fare
average_fare = total_fare / fare_count if fare_count > 0 else 0

# Printing the results
print(f"Total Fare: ${total_fare:,.2f}")
print(f"Average Fare: ${average_fare:,.2f}")
print(f"Maximum Trip Distance: {max_trip_miles:.2f} miles")