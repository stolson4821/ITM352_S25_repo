test_cases = [
    [1, 2],                           # Fewer than 5 elements
    [1, 2, 3, 4, 5],                  # Exactly 5 elements
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Exactly 10 elements
    list(range(15))                    # More than 10 elements
]

for test_list in test_cases:
    length = len(test_list)
    print(f"Testing list of length {length}: {test_list}")
    
    if length < 5:
        print("The list has fewer than 5 elements.")
    elif 5 <= length <= 10:
        print("The list has between 5 and 10 elements.")
    else:
        print("The list has more than 10 elements.")
    print("-" * 40)