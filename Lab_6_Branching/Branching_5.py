def determine_movie_price(age, weekday, matinee):
    normal_price = 14
    senior_price = 8 if age >= 65 else normal_price
    tuesday_price = 10 if weekday == "Tuesday" else normal_price
    matinee_price = 5 if age >= 65 else 8 if matinee else normal_price
    
    return min(senior_price, tuesday_price, matinee_price)

# Example input
age = int(input("Enter age: "))
weekday = input("Enter weekday: ")
matinee = input("Is it a matinee? (yes/no): ").strip().lower() == "yes"

# Determine and print the price
price = determine_movie_price(age, weekday, matinee)
print(f"Age: {age}, Weekday: {weekday}, Matinee: {matinee}, Price: ${price}")
