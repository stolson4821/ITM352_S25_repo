def fahrenheit_to_celsius():
    user_input = int(input("Input a temperature in Fahrenheit:"))
    celcius = round((user_input - 32) * (5 / 9) , 0)
    print(f"Your temperature of {user_input} degrees F. is equal to {celcius} degrees C.")

fahrenheit_to_celsius()