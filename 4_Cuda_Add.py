from numba import cuda
import numpy as np


# ----------------------------------------
# VECTOR ADDITION KERNEL
# ----------------------------------------
@cuda.jit
def vector_add(a, b, c):

    # Get thread index
    i = cuda.threadIdx.x

    # Boundary check
    if i < c.size:

        # Add vector elements
        c[i] = a[i] + b[i]


# ----------------------------------------
# USER INPUT
# ----------------------------------------

# Size of vectors
n = int(input("Enter size of vectors: "))

print("Enter elements of first vector:")
a_list = []
for i in range(n):
    val = int(input(f"a[{i}] = "))
    a_list.append(val)

print("Enter elements of second vector:")
b_list = []
for i in range(n):
    val = int(input(f"b[{i}] = "))
    b_list.append(val)

# Convert to NumPy arrays
a = np.array(a_list)
b = np.array(b_list)

# Output vector
c = np.zeros(n)

# Launch kernel
vector_add[1, n](a, b, c)

# Output result
print("\nVector Addition Result:")
print(c)
