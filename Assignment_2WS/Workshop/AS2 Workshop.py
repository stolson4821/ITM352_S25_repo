# Sales Analytics Dashboard
import pandas as pd
import pyarrow
import time
import os

# Utility function to export DataFrame to Excel
def export_to_excel(df, filename):
    try:
        df.to_excel(f"{filename}.xlsx", index=False)
        print(f"Data exported to {filename}.xlsx successfully.")
    except Exception as e:
        print(f"Failed to export data to Excel: {e}")

# Function to get user selection from a list of options
def get_user_selection(options, prompt):
    print(prompt)
    for i, option in enumerate(options):
        print(f"{i+1}. {option}")
    choice = input("Enter the number(s) of your choice(s), separated by commas: ").strip()
    try:
        selected = [options[int(i) - 1] for i in choice.split(',')] if choice else []
        return selected
    except (ValueError, IndexError):
        print("Invalid input. Please try again.")
        return []

# Step 3: Function to display initial rows
def display_initial_rows(data):
    try:
        num_rows = input("How many rows would you like to see? (Enter a number, 'all', or leave blank to skip): ").strip()
        if num_rows.lower() == 'all':
            print(data)
        elif num_rows.isdigit():
            print(data.head(int(num_rows)))
        elif num_rows == '':
            print("Skipping preview.")
        else:
            print("Invalid input. Please enter a number or 'all'.")
    except Exception as e:
        print(f"Error displaying rows: {e}")

# Step 4: Display employees by region (pivot table)
def sales_by_region_order_type(data):
    try:
        if all(col in data.columns for col in ['sales_region', 'order_type', 'employee_id']):
            pivot = data.pivot_table(index='sales_region', columns='order_type', values='employee_id', aggfunc=pd.Series.nunique)
            print(pivot)
            export = input("Export this pivot table to Excel? (y/n): ").strip().lower()
            if export == 'y':
                filename = input("Enter the filename (without extension): ").strip()
                export_to_excel(pivot, filename)
            return pivot
        else:
            print("Error: Required columns ('sales_region', 'order_type', 'employee_id') are missing.")
    except Exception as e:
        print(f"Error generating pivot table: {e}")

# Step 5: Generate a custom pivot table
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

# Step 6: Exit the program
def exit_program(data):
    print("Exiting the program.")
    exit()

# Display menu function
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

# Main function to initiate the dashboard
def main():
    url = 'https://drive.google.com/uc?export=download&id=1Fv_vhoN4sTrUaozFPfzr0NCyHJLIeXEA'
    create_test_csv(url)
    data = load_csv("sales_data_test.csv")
    menu_options = [display_initial_rows, sales_by_region_order_type, generate_custom_pivot_table, exit_program]
    while True:
        choice = display_menu()
        if choice is not None:
            try:
                menu_options[choice](data)
            except Exception as e:
                print(f"Error executing the selected option: {e}")
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
