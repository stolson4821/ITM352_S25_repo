# File name to save the JSON data
filename = "guitar_questions.json"

# Save the dictionary to a JSON file
with open(filename, mode="w") as json_file:
    json.dump(guitar_questions, json_file, indent=4)  # Remove the extra period after 'guitar_questions'

print(f"Questions saved to {filename}")