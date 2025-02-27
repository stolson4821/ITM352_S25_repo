my_tuple = ("hello", 10, "goodbye", 3, "goodnight", 5)
new_value = input("Enter a value to append to the tuple: ")
try:
    my_tuple.append(new_value)
except AttributeError as e:
    print(f"Error: Cannot append to a tuple. {e}")