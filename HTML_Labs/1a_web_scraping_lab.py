import ssl
import urllib.request

#bypass SSL verification 
ssl._create_default_https_context = ssl._create_unverified_context

url = "https://data.cityofchicago.org/Historic-Preservation/Landmark-Districts/zidz-sdfj/about_data"

with urllib.request.urlopen(url) as response:
    print(response)
   
