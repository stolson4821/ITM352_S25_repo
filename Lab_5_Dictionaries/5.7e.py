trip_dict = {1.1: 6.25, 0.8: 5.25, 2.5: 10.50, 2.6: 8.05}
trips_list = [
    {'duration': 1.1, 'fare': 6.25},
    {'duration': 0.8, 'fare': 5.25},
    {'duration': 2.5, 'fare': 10.50},
    {'duration': 2.6, 'fare': 8.05}
]

# Print the duration and fare of the 3rd trip
print(f"Duration of 3rd trip: {trips_list[2]['duration']} hours")
print(f"Cost of 3rd trip:  ${trips_list[2]['fare']}")

# Convert trip_dict to a list of dictionaries
trips_list = [{'duration': duration, 'fare': fare} for duration, fare in trip_dict.items()]
