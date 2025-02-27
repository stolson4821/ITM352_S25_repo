#Write Python code that uses a Python for-statement to create 
# a list of elements that are the odd numbers between 1 and 50.

odd_numbers = []
for num in range(1, 50):
    if num % 2 != 0:
        odd_numbers.append(num)
print(odd_numbers)
