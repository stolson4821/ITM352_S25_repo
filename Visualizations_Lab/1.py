import matplotlib.pyplot as plt

# a. Define lists of x and y values (first dataset)
x1 = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]

# d. Define a second set of x and y values
x2 = [1.5, 2.3, 3.7, 4.1, 5.5]
y2 = [1, 7, 3, 8, 4]

# b. Plot the first set as a line graph
plt.plot(x1, y1, label='Line 1', color='blue')

# c. Plot the first set as a scatter plot
plt.scatter(x1, y1, label='Scatter 1', color='blue', marker='o')

# d. Add the second set as a line graph
plt.plot(x2, y2, label='Line 2', color='red', linestyle='--')

# e. Add a title and axis labels
plt.title('Line and Scatter Plot Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Add legend for clarity
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
