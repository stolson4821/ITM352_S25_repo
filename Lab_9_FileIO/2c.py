import csv
import os

# File name
filename = "FileIOlab.csv"

# Check if the file exists and is readable
if os.path.exists(filename) and os.path.isfile(filename) and os.access(filename, os.R_OK):
    # Get file information
    file_size = os.path.getsize(filename)  # File size in bytes
    absolute_path = os.path.abspath(filename)  # Full file path
    last_modified = os.path.getmtime(filename)  # Last modified timestamp

    # Print file details
    print(f"File Information for '{filename}':")
    print(f" - Size: {file_size} bytes")
    print(f" - Absolute Path: {absolute_path}")
    print(f" - Last Modified Timestamp: {last_modified}\n")

    # List to store Annual_Salary values
    salaries = []

    # Read the CSV file
    with open(filename, mode="r", newline="") as file:
        reader = csv.DictReader(file)  # Read as a dictionary for easier column access
        for row in reader:
            try:
                # Convert salary to integer and store
                salaries.append(int(row["Annual_Salary"]))
            except ValueError:
                print(f"Skipping invalid salary value: {row['Annual_Salary']}")  # Handle errors gracefully

    # Ensure there are salaries to calculate statistics
    if salaries:
        # Calculate required statistics
        average_salary = sum(salaries) / len(salaries)
        max_salary = max(salaries)
        min_salary = min(salaries)

        # Display results in clea