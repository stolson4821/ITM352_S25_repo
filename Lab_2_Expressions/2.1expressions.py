##Chat GPT in class
# Ask user to enter a qhole number between 1 and 100
gpt_input = ("3")

#Convert input to integer and square the number
gpt_number = int(gpt_input)
gpt_squared_number = int(gpt_number ** 2)

# Print results
print(f"The square of {gpt_number} is {gpt_squared_number}")

##By hand in class
#With only one variable
user_input = int(input("Enter a number 1-100"))
print("The square of " + str(user_input) + " is " +  str(user_input ** 2))
