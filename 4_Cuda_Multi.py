from numba import cuda
import numpy as np

# ----------------------------------------
# MATRIX MULTIPLICATION KERNEL
# ----------------------------------------
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


# Input matrices
a = np.array([[1, 2],
              [3, 4]])

b = np.array([[5, 6],
              [7, 8]])

# Output matrix
c = np.zeros((2, 2))

# Threads per block
threads_per_block = (2, 2)

# Blocks per grid
blocks_per_grid = (1, 1)

# Launch kernel
matrix_mul[blocks_per_grid, threads_per_block](a, b, c)

print("\nMatrix Multiplication Result:")
print(c)
