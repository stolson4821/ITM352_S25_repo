import requests
from bs4 import BeautifulSoup
import re

# URL of the mortgage rates page
url = 'https://www.hicentral.com/hawaii-mortgage-rates.php'

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table containing mortgage rates
    table = soup.find('table')

    # Iterate over each row in the table
    lender_name = None
    term_type = None
    interest_rate = None

    for row in table.find_all('tr')[1:]:  # Skip the header row
        cells = row.find_all('td')
        if len(cells) >= 4:
            # Extract lender name, term/type, interest rate, and points
            lender_name_cell = cells[0].get_text(strip=True)
            if lender_name_cell:
                # Remove the NMLS# and phone number from lender names
                lender_name_clean = re.sub(r'NMLS#\S+', '', lender_name_cell)  # Remove NMLS#
                lender_name_clean = re.sub(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', '', lender_name_clean)  # Remove phone numbers
                lender_name_clean = lender_name_clean.strip()
                lender_name = lender_name_clean

            term_type_cell = cells[1].get_text(strip=True)
            interest_rate_cell = cells[2].get_text(strip=True)

            if term_type_cell:
                term_type = term_type_cell
            if interest_rate_cell:
                interest_rate = interest_rate_cell

            # If we have both a term type and interest rate, print the result
            if term_type and interest_rate:
                print(f'Lender: {lender_name}')
                print(f'{term_type} Interest Rate: {interest_rate}')
                print('-' * 40)
else:
    print(f'Failed to retrieve the page. Status code: {response.status_code}')