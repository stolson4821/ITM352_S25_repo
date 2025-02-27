my_list = [1, "apple", 3.14, True, 42, "banana", None, [5, 6], {"key": "value"}]

length = len(my_list)

if length < 5:
    print("The list has fewer than 5 elements.")
elif 5 <= length <= 10:
    print("The list has between 5 and 10 elements.")
else:
    print("The list has more than 10 elements.")