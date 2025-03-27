import numpy as np

# Creating the NumPy array
percentile_income = np.array([
    (10, 14629),
    (20, 25600),
    (30, 37002),
    (40, 50000),
    (50, 63179),
    (60, 79542),
    (70, 100162),
    (80, 130000),
    (90, 184292)
])

# Getting the dimensions and number of elements
print("Array Shape:", percentile_income.shape)
print("Number of Elements:", percentile_income.size)