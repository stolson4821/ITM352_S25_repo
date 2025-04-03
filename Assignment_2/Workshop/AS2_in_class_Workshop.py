import pandas as pd
import numpy as np
import pyarrow
import os
import urllib.request
import ssl
import time

# Define the direct download link from Google Drive
url = "https://drive.google.com/uc?export=download&id=1Fv_vhoN4sTrUaozFPfzr0NCyHJLIeXEA"

def load_csv(filepath):
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

def display_initial_rows(data):
    """
    Prompts the user to specify how many initial rows of the DataFrame they would like to see.
    """
    while True:
        user_input = input("Enter the number of rows to display, 'all' for all rows, or press Enter to skip: ").strip().lower()
        
        if user_input == '':
            print("Skipping preview.")
            return
        elif user_input == 'all':
            print(data)
            return
        elif user_input.isdigit():
            num_rows = int(user_input)
            if 0 < num_rows <= len(data):
                print(data.head(num_rows))
                return
            else:
                print(f"Please enter a number between 1 and {len(data)}.")
        else:
            print("Invalid input. Please enter a valid number, 'all', or leave empty to skip.")

def main():
    # Download the CSV file from Google Drive and save it locally
    try:
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
    df_loaded = load_csv(csv_path)
    if df_loaded is not None:
        print(df_loaded.head())

        # Continuously prompt the user to display initial rows of the DataFrame
        while True:
            display_initial_rows(df_loaded)

if __name__ == "__main__":
    main()
