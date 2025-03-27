import pandas as pd

# Lists of individuals' names, ages, and genders
names = ["Joe", "Jaden", "Max", "Sidney", "Evgeni", "Taylor", "Pia", "Luis", "Blanca", "Cyndi"]
ages = [25, 30, 22, 35, 28, 40, 50, 18, 60, 45]
gender = ["M", "M", "M", "F", "M", "F", "F", "M", "F", "F"]

# Creating a DataFrame with names as index
df = pd.DataFrame(list(zip(ages, gender)), index=names, columns=["Age", "Gender"])

mean_by_gender = df.groupby("Gender").mean().round(3)

# Printing only the mean statistics for each gender
print("Mean Statistics by Gender:")
print(mean_by_gender)
