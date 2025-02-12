first_name = input("Enter your first name: ")
middle_initial = input("Now your middle initial: ")
last_name = input("Finally your last name:")

#Part A
#full_name = first_name + " " + middle_initial + " " + last_name
#print("Full name:", full_name)

#Part B
#full_name = f"{first_name} {middle_initial} {last_name}"
#print(f"Your full name is ", full_name)

#Part C
#full_name = "%s %s %s" % (first_name, middle_initial, last_name)
#print("Your full name is {full_name}.")

#Part D
#full_name = format(first_name + middle_initial + last_name)
#print(full_name)

#Part E
#full_name = " ".join([first_name, middle_initial, last_name])
#print(full_name)

#Part F
name_parts = [first_name, middle_initial, last_name]
full_name = "{} {} {}".format(*name_parts)
print(full_name)