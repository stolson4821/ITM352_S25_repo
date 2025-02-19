trip_durations = [1.1, 0.8, 2.5, 2.6]
trip_fares = [6.25, 5.25, 10.50, 8.05]
#converted both to lists for ease of use and possible mutilation
trips ={
    'miles' : trip_durations,
    'fares' : trip_fares
}
trip_num = int(input('What trip do you want to look up:'))#enter 3
print(f"Duration is : {trips['miles'][trip_num - 1]} miles") 
print(f"Fare is : ${trips['fares'][trip_num - 1]}")