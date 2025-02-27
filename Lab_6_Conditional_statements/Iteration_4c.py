def classify_fare(fare):
    return f"${fare}: This fare is high!" if fare > 12 else f"${fare}: This fare is low."
test_fares = [8.60, 5.75, 13.25, 21.21]
for fare in test_fares:
    print(classify_fare(fare))