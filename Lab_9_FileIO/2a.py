import csv

# File name
filename = "FileIOlab.csv"

# List to store Annual_Salary values
salaries = []

# Read the CSV file
with open(filename, newline="") as file:
    reader = csv.DictReader(file) 
# Read as a dictionary for easier column access
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

# Display results in clear language
    print(f"Average Annual Salary: ${average_salary:,.2f}")
    print(f"Maximum Annual Salary: ${max_salary:,.2f}")
    print(f"Minimum Annual Salary: ${min_salary:,.2f}")
else:
    print("No valid salary data found.")
