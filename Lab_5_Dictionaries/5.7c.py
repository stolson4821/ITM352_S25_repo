trip_durations = [1.1, 0.8, 2.5, 2.6]
trip_fares = [6.25, 5.25, 10.50, 8.05]

# Create dictionary using zip()
trip_dict = dict(zip(trip_durations, trip_fares))

# Print duration and cost of the 3rd trip
third_trip_duration = trip_durations[2]  # Index 2 (3rd element)
third_trip_cost = trip_dict[third_trip_duration]

print(f"Duration of 3rd trip: {third_trip_duration} hours")
print(f"Cost of 3rd trip: ${third_trip_cost}")
