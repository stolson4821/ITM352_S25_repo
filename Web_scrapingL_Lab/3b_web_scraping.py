import requests
from bs4 import BeautifulSoup
import re

# URL of the ITM faculty directory
url = "https://shidler.hawaii.edu/itm/people"

# Fetch the HTML content
response = requests.get(url)
response.raise_for_status()  # Raise an error for bad status codes

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all <a> tags with href attributes matching the pattern '/itm/directory/{faculty-name}'
faculty_links = soup.find_all('a', href=re.compile(r'^/itm/directory/[\w-]+$'))

# Use a set to store unique faculty names
faculty_names = set()

for link in faculty_links:
    href = link['href']
    # Extract the faculty name from the href
    faculty_name = href.split('/')[-1]
    # Replace hyphens with spaces and capitalize each word
    formatted_name = faculty_name.replace('-', ' ').title()
    faculty_names.add(formatted_name)

# Sort names alphabetically
sorted_names = sorted(faculty_names)

# Print the list of faculty names and the total count
print("Faculty Names:")
for name in sorted_names:
    print(name)
print(f"\nTotal number of unique faculty members found: {len(sorted_names)}")