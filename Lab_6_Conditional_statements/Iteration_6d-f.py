my_tuple = ("hello", 10, "goodbye", 3, "goodnight", 5)
new_value = input("Enter a value to append to the tuple: ")

##Part D
#my_tuple = my_tuple + (new_value,)

##Part E
#my_tuple = (*my_tuple, new_value) 
#print("Updated tuple:", my_tuple)

##Part F
temp_list = list(my_tuple)
temp_list.append(new_value)
my_tuple = tuple(temp_list)

print("Updated tuple:", my_tuple)
