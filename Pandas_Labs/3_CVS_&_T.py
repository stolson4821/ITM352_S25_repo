import pandas as pd

data = {
   'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
   'Age': [25, 30, 35, 40, 22],
   'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
   'Salary': [70000, 80000, 120000, 90000, 60000]
}
#Create data frame from available data.
df = pd.DataFrame(data)

#ensure list starts at 1 and not 0.
df.index = range(1, len(df) + 1)

#print it out.
print(df)