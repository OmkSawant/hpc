from numba import cuda
import numpy as np

# MATRIX MULTIPLICATION KERNEL

@cuda.jit
def matrix_mul(a, b, c):

    # Get row and column index
    row, col = cuda.grid(2)

    # Boundary check
    if row < c.shape[0] and col < c.shape[1]:

        total = 0

        # Matrix multiplication
        for k in range(a.shape[1]):

            total += a[row][k] * b[k][col]

        # Store result
        c[row][col] = total


# USER INPUT

# Matrix size
rows_a = int(input("Enter number of rows for Matrix A: "))
cols_a = int(input("Enter number of columns for Matrix A: "))

rows_b = int(input("Enter number of rows for Matrix B: "))
cols_b = int(input("Enter number of columns for Matrix B: "))

# Check multiplication condition
if cols_a != rows_b:
    print("Matrix multiplication not possible!")
    exit()

# Input Matrix A
print("\nEnter elements of Matrix A:")
a_list = []

for i in range(rows_a):
    row = []
    for j in range(cols_a):
        val = int(input(f"A[{i}][{j}] = "))
        row.append(val)
    a_list.append(row)

# Input Matrix B
print("\nEnter elements of Matrix B:")
b_list = []

for i in range(rows_b):
    row = []
    for j in range(cols_b):
        val = int(input(f"B[{i}][{j}] = "))
        row.append(val)
    b_list.append(row)

# Convert to NumPy arrays
a = np.array(a_list)
b = np.array(b_list)

# Output matrix
c = np.zeros((rows_a, cols_b))

# Threads per block
threads_per_block = (16, 16)

# Blocks per grid
blocks_per_grid_x = (rows_a + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (cols_b + threads_per_block[1] - 1) // threads_per_block[1]

blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Launch kernel
matrix_mul[blocks_per_grid, threads_per_block](a, b, c)

# Print result
print("\nMatrix Multiplication Result:")
print(c)
