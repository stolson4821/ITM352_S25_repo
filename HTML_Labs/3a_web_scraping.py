import requests
from bs4 import BeautifulSoup

# URL to inspect
url = "https://shidler.hawaii.edu/itm/people"

# Make a GET request to fetch the raw HTML content
response = requests.get(url)

# Parse the HTML using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Print the type of object
print("Type of parsed object:", type(soup))

# Print the first few lines of the formatted HTML
print("\nFirst few lines of HTML:")
# printing only the first 500 characters
print(soup.prettify()[:500])  
