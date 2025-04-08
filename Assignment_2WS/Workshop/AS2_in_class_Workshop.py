import pandas as pd
import numpy as np
import pyarrow
import os
import urllib.request
import ssl
import time
import sys

# Define the direct download link from Google Drive
url = "https://drive.google.com/uc?export=download&id=1Fv_vhoN4sTrUaozFPfzr0NCyHJLIeXEA"

#Loads Sales.csv creates dataframe displays rows & columns
def load_sales_to_df(filepath):
    """
    Loads a CSV file into a DataFrame with error handling and performance tracking.
    Uses the pyarrow backend, skips bad rows, and converts dates to a more useful format.
    """
    required_columns = ['quantity', 'unit_price']
    
    try:
        print(f"Loading CSV file from: {filepath}...")

        # Start timing the loading process
        start_time = time.time()

        # Read the CSV file with pyarrow engine and skip bad lines
        df = pd.read_csv(filepath, engine='pyarrow', on_bad_lines="skip", parse_dates=True)

        # Calculate loading time
        load_time = time.time() - start_time
        print(f"File loaded successfully in {load_time:.2f} seconds.")

        # Display basic information about the DataFrame
        print(f"Number of rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")

        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}. Some analytics may not function correctly.")
        else:
            print("All necessary columns are present.")

        # Convert date columns to a more useful format (if present)
        for col in df.select_dtypes(include=['datetime']):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"Converted date column: {col}")

        return df

    except Exception as e:
        print(f"File loading error: {e}")
        return None

#Exit program function
def exit_program():
    sys.exit("Exiting Program.")

#Row selector and display function
def display_initial_rows(data):
    """
    Prompts the user to specify how many initial rows of the DataFrame they would like to see.
    This function handles different input types and gracefully manages invalid entries.
    """
    while True:
        # Prompt user for input on how many rows to display
        user_input = input("Enter the number of rows to display, 'all' for all rows, or press Enter to skip: ").strip().lower()
        
        # If user presses Enter without typing anything, skip the preview
        if user_input == '':
            print("Skipping preview.")
            return
        
        # If user types 'all', display all rows of the DataFrame
        elif user_input == 'all':
            print(data)
            return
        
        # If user enters a number, attempt to display that many rows
        elif user_input.isdigit():
            num_rows = int(user_input)  # Convert the input to an integer
            
            # Check if the entered number is within a valid range (1 to the number of rows in the DataFrame)
            if 0 < num_rows <= len(data):
                print(data.head(num_rows))  # Display the first 'num_rows' rows
                return
            else:
                # Inform the user if the number is out of valid range
                print(f"Please enter a number between 1 and {len(data)}.")
        
        # If the input is neither 'all', a valid number, nor an empty input, display an error message
        else:
            print("Invalid input. Please enter a valid number, 'all', or leave empty to skip.")

#Pivots sales by region order type
def sales_by_region_order_type(data):
    print("Sales by region order type:")
    try:
        # Create the pivot table with the count of unique employees for each region and order type
        pivot_table = pd.pivot_table(
            data,
            index='sales_region',       # Rows: Sales region
            columns='order_type',       # Columns: Order type
            values='employee_id',       # Values: Employee ID to count unique employees
            aggfunc=pd.Series.nunique,  # Aggregation: Count of unique employees
            fill_value=0                # Fill missing values with 0
        )

        # Display the pivot table
        print(pivot_table)

        # Ask the user if they want to export the pivot table to an Excel file
        export = input("Would you like to export the pivot table to an Excel file? (yes/no): ").strip().lower()
        if export == 'yes':
            filename = input("Enter the filename (without extension): ").strip()
            pivot_table.to_excel(f"{filename}.xlsx", index=True)
            print(f"Pivot table exported successfully as {filename}.xlsx")
        else:
            print("Export skipped.")

        # Return the pivot table for potential comparison or further use
        return pivot_table

    except Exception as e:
        print(f"Error while generating pivot table: {e}")
        
#Generate custom pivot table
def generate_custom_pivot_table(data):
    print("Custom table")
    row_options = list(data.columns)
    rows = get_user_selection(row_options, "Select rows:")
    col_options = [col for col in row_options if col not in rows]
    cols = get_user_selection(col_options, "Select columns:")
    value_options = list(data.select_dtypes(include=['number']).columns)
    values = get_user_selection(value_options, "Select values:")
    agg_options = ['sum', 'mean', 'count']
    agg_func = get_user_selection(agg_options, "Select aggregation function:")[0] if get_user_selection(agg_options, "Select aggregation function:") else 'sum'
    pivot_table = pd.pivot_table(data, index=rows, columns=cols if cols else None, values=values, aggfunc=agg_func)
    print(pivot_table)
    return pivot_table 

def main():
    try:  # Download the CSV file from Google Drive and save it locally
        # Create an SSL context to ignore certificate verification
        context = ssl._create_unverified_context()

        # Download the file from the URL
        response = urllib.request.urlopen(url, context=context)

        # Read CSV with error handling and skip bad lines
        df = pd.read_csv(response, engine='pyarrow', on_bad_lines="skip")
        print(df.head())

        # Get the current directory of the script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "sales_data_test.csv")

        # Save the DataFrame to a CSV file in the same directory
        df.to_csv(csv_path, index=False)
        print(f'Data saved successfully to {csv_path}')

    except Exception as e:
        print(f'File reading error: {e}')

    # Load the CSV file using the newly created function
    df_loaded = load_sales_to_df(csv_path)
    if df_loaded is not None:
        print(df_loaded.head())

        # Continuously prompt the user to display initial rows of the DataFrame
        while True:
            display_initial_rows(df_loaded)
            exit_program("Exiting Program")

#display Main Menu
def display_menu():
    menu_options = (
        ("Show the first n rows of sales data", display_initial_rows),
        ("Show the number of employees by region", sales_by_region_order_type),
        ("Generate a custom pivot table", generate_custom_pivot_table),
        ("Exit the program", exit_program)
    )
    for idx, (desc, _) in enumerate(menu_options, start=1):
        print(f"{idx}. {desc}")
    choice = input("Select an option: ").strip()
    return int(choice) - 1 if choice.isdigit() and 0 < int(choice) <= len(menu_options) else None

if __name__ == "__main__":
    main()
