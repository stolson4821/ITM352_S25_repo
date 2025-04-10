import ssl
import urllib.request

#bypass SSL verification 
ssl._create_default_https_context = ssl._create_unverified_context

url = "https://data.cityofchicago.org/Historic-Preservation/Landmark-Districts/zidz-sdfj/about_data"

with urllib.request.urlopen(url) as response:
    for line in response:
        decoded_line = line.decode('utf-8')
        if "<title>" in decoded_line:
            print(decoded_line.strip())
