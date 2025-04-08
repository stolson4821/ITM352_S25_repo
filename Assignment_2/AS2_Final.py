import pandas as pd
import pyarrow as py
import os
import time
import requests
from io import StringIO

# Function to download the CSV file from Google Drive using the file ID
    #this was an AI assisted function cause i was having SSL issues getting the file with how we were doing it in class. 
def download_sales_data(file_id):
    print("Downloading sales data from Google Drive...")
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.get(download_url)
        response.raise_for_status()  # Check if the request was successful
        return StringIO(response.text)  # Return CSV data as a file-like object
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        exit()

# Function to load sales data from a CSV file and handle errors (in class)
def load_sales_data():
    file_id = "1Fv_vhoN4sTrUaozFPfzr0NCyHJLIeXEA"  # Google Drive file ID
    print("Loading sales data...")
    start_time = time.time()
    
    # Download the data with time it took. (in class)
    csv_data = download_sales_data(file_id)
    
    try:
        data = pd.read_csv(csv_data)
        # Replace missing values with zero 
        data = data.fillna(0)
        load_time = time.time() - start_time
        print(f"Data loaded successfully in {load_time:.2f} seconds.")
        print(f"Rows: {len(data)}, Columns: {len(data.columns)}")
        print("Available columns:", data.columns.tolist())
        
        # Check if required columns are present
        required_columns = ['sales_region', 'order_type', 'employee_id', 'sales', 'customer_type', 'product_category', 'quantity', 'unit_price']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"Warning: Missing columns for analysis: {', '.join(missing_columns)}")
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

# Define the total sales function (quantity * unit_price) (no AI Used)
def calculate_total_sales(data):
    if 'quantity' in data.columns and 'unit_price' in data.columns:
        # Calculate total sales as quantity * unit_price for each row
        data['total_sales'] = data['quantity'] * data['unit_price']
    else:
        print("Error: Missing 'quantity' or 'unit_price' columns for total sales calculation.")
        data['total_sales'] = 0
    return data

# Export DataFrame to Excel (In class)(no AI used)
def export_to_excel(df, filename):
    try:
        print("Exporting DataFrame:")
        print(df.head())
        print(f"DataFrame shape: {df.shape}")
        if df.empty:
            print("The DataFrame is empty. No data to export.")
            return
        df.to_excel(f"{filename}.xlsx", index=False)
        print(f"Data exported successfully to {filename}.xlsx")
    except Exception as e:
        print(f"Error exporting data: {e}")

# Get user selection from a list of options (in class) (AI assisted only help to enumerate options)
def get_user_selection(options, prompt):
    print(prompt)
    for i, option in enumerate(options):
        print(f"{i+1}. {option}")
    choice = input("Enter number(s), separated by commas: ").strip()
    try:
        selected = [options[int(i) - 1] for i in choice.split(',')] if choice else []
        return selected
    except:
        print("Invalid input.")
        return []

# 1 Show the first n rows of sales data (in class) (No AI Used)
def display_initial_rows(data):
    num_rows = input(f"How many rows to display? (1 to {len(data)}, 'all' for all, or press Enter to skip): ").strip()
    if num_rows == 'all':
        print(data)
    elif num_rows.isdigit() and 1 <= int(num_rows) <= len(data):
        print(data.head(int(num_rows)))
    elif not num_rows:
        print("Skipping row preview.")
    else:
        print("Invalid input. Please enter a valid number or 'all'.")

# 2 Predefined pivot table: Total sales by region and order_type (In class partial) (No AI Used)
def total_sales_by_region_order_type(data):
    try:
        # Calculate total sales first
        data = calculate_total_sales(data)
        
        # Create pivot table by region (rows) and order type (columns)
        pivot = data.pivot_table(index='sales_region', columns='order_type', values='total_sales', aggfunc='sum')
        
        # Print the pivot table
        print(pivot)
        
        # Ask if user wants to export to Excel
        export = input("Would you like to export the results to Excel? (y/n): ").strip().lower()
        if export == 'y':
            filename = input("Name the File: ").strip()
            export_to_excel(pivot.reset_index(), filename)
    except Exception as e:
        print("Error:", e)

# 3 Average sales by region, state, and order_type (no AI Used) This was a headache for me. once i had a working function i had GPT remove redundant lines. It used to be dragged out to line 170. haha
def average_sales_by_region_state_order_type(data):
    try:
        # Calculate total sales
        data = calculate_total_sales(data)

        # Create pivot table
        pivot = data.pivot_table(
            index=['customer_state', 'order_type'],
            columns='sales_region',
            values='total_sales',
            aggfunc='mean'
        )

        # Check if pivot table is empty or has valid data
        print("Pivot table preview:")
        print(pivot.head())  # Print the first few rows of the pivot table
        print("Pivot table shape:", pivot.shape)

        # Fill NaN values with 0
        pivot.fillna(0, inplace=True)

        # Flatten the MultiIndex columns by joining the levels into a single string
        pivot.columns = ['_'.join(map(str, col)).strip() if isinstance(col, tuple) else str(col) for col in pivot.columns]

        # Show pivot table again after filling NaN values and flattening columns
        print("Pivot table after flattening columns:")
        print(pivot.head())  # Print the first few rows of the pivot table
        print("Pivot shape after flattening columns:", pivot.shape)

        # Export to Excel if user agrees
        export = input("Would you like to export the results to Excel? (y/n): ").strip().lower()
        if export == 'y':
            filename = input("Name the File: ").strip()
            print("Exporting pivot table...")
            print("Pivot shape before export:", pivot.shape)
            # Ensure we reset index and check data before exporting
            export_to_excel(pivot.reset_index(), filename)

    except Exception as e:
        print(f"Error: {e}")

# 4 Create a custom pivot table based on user input (AI used only involved to flatten multi indexed columns and fill Nan values with 0)
def generate_custom_pivot_table(data):
    rows = get_user_selection(list(data.columns), "Select rows:")
    cols = get_user_selection(list(data.columns), "Select columns:")
    values = get_user_selection(list(data.select_dtypes(include=['number']).columns), "Select values:")
    agg_func = get_user_selection(["sum", "mean", "count"], "Select aggregation function:")

    try:
        # Calculate total sales if not already calculated
        data = calculate_total_sales(data)

        # Create pivot table
        pivot = pd.pivot_table(data, index=rows, columns=cols, values=values, aggfunc=agg_func[0])

        # Check pivot table before export
        print(pivot)
        print("Pivot shape before export:", pivot.shape)

        # Flatten MultiIndex columns if needed
        if isinstance(pivot.columns, pd.MultiIndex):
            pivot.columns = ['_'.join(map(str, col)).strip() for col in pivot.columns]

        # Fill NaN values with 0
        pivot.fillna(0, inplace=True)

        # Round the pivot table to 2 decimal places
        pivot = pivot.round(2)

        # Show the pivot table after flattening and filling NaN values
        print("Pivot table after flattening columns, filling NaN values, and rounding to 2 decimal places:")
        print(pivot)

        # Export to Excel if user agrees
        export = input("Would you like to export the results to Excel? (y/n): ").strip().lower()
        if export == 'y':
            filename = input("Name the File: ").strip()
            print("Exporting pivot table...")
            # Ensure we reset index and check data before exporting
            export_to_excel(pivot.reset_index(), filename)

        return pivot  # Return the pivot table
    except Exception as e:
        print(f"Error generating custom pivot table: {e}")
        return None  # Return None in case of an error

#5 Compare exported Pivot tables (No AI used other than to look up the propper steps to complete this step by asking it "what steps do i need to take to complete this tak in python")
def compare_pivot_tables_side_by_side(data):
    try:
        # Create the first pivot table using the custom pivot table generation function
        print("Creating the first pivot table...")
        pivot1 = generate_custom_pivot_table(data)  # Now returns the pivot table
        if pivot1 is None:
            print("Error creating first pivot table. Exiting.")
            return

        # Create the second pivot table using the custom pivot table generation function
        print("\nCreating the second pivot table...")
        pivot2 = generate_custom_pivot_table(data)  # Now returns the pivot table
        if pivot2 is None:
            print("Error creating second pivot table. Exiting.")
            return
        
        # Display both pivot tables side by side for comparison
        comparison = pd.concat([pivot1, pivot2], axis=1, keys=["Pivot1", "Pivot2"])
        print("\nComparison of Pivot Table 1 and Pivot Table 2:")
        print(comparison)
        
        # Optionally, allow the user to export the comparison
        export = input("Would you like to export the comparison to Excel? (y/n): ").strip().lower()
        if export == 'y':
            filename = input("Name the File: ").strip()
            with pd.ExcelWriter(f"{filename}_comparison.xlsx") as writer:
                # Export both pivot tables and the comparison
                pivot1.to_excel(writer, sheet_name="Pivot1")
                pivot2.to_excel(writer, sheet_name="Pivot2")
                comparison.to_excel(writer, sheet_name="Comparison")
            print(f"Comparison exported to {filename}_comparison.xlsx")
    
    except Exception as e:
        print(f"Error comparing pivot tables: {e}")      

# 6 Exit the program
def exit_program(_=None):
    print("Goodbye!")
    exit()

# Display the menu and get user choice (In class No AI used)
def display_menu():
    options = [
        ("Show rows", display_initial_rows),
        ("Total sales by region and order_type", total_sales_by_region_order_type),
        ("Average sales by region, state, and sale type", average_sales_by_region_state_order_type),
        ("Create a custom pivot table", generate_custom_pivot_table),
        ("Compare exported Pivot Tables", compare_pivot_tables_side_by_side),  # Corrected function name and added missing comma
        ("Exit", exit_program)
    ]
    while True:
        # Display options
        for i, (desc, _) in enumerate(options):
            print(f"{i+1}. {desc}")
        
        # Get user input and validate it
        try:
            choice = int(input("Choose an option: "))
            if 1 <= choice <= len(options):
                return choice - 1  # Valid choice, return index
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Main function to run the dashboard (in class No AI used)
def main():
    # Load data
    data = load_sales_data()
    
    # Calculate total sales if not already present
    data = calculate_total_sales(data)
    
    functions = [
        display_initial_rows,
        total_sales_by_region_order_type,
        average_sales_by_region_state_order_type,
        generate_custom_pivot_table,
        compare_pivot_tables_side_by_side,
        exit_program
    ]
    
    while True:
        choice = display_menu()
        if choice is not None and 0 <= choice < len(functions):
            functions[choice](data)
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
