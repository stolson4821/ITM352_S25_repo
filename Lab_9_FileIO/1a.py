# Open the file in read mode
file_obj = open('names.txt', 'r')

# Display the data type returned from open()
print(type(file_obj))

#Close the file
file_obj.close()