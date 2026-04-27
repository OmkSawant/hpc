import numpy as np
import time
from multiprocessing import Pool, cpu_count

# ---------- Sequential Vector Addition ----------
def sequential_add(A, B):
    C = []
    for i in range(len(A)):
        C.append(A[i] + B[i])
    return C

# ---------- Parallel Worker ----------
def add_pair(pair):
    return pair[0] + pair[1]

# ---------- Parallel Vector Addition ----------
def parallel_add(A, B):
    with Pool(cpu_count()) as p:
        C = p.map(add_pair, zip(A, B))
    return C

# ---------- Main ----------
if __name__ == "__main__":

    choice = input("Enter '1' for manual input or '2' for random vectors: ")

    if choice == '1':
        A = list(map(int, input("Enter elements of vector A: ").split()))
        B = list(map(int, input("Enter elements of vector B: ").split()))

        if len(A) != len(B):
            print("Error: Vectors must be of same length!")
            exit()
    else:
        N = int(input("Enter size of vectors: "))
        A = np.random.randint(0, 100, N)
        B = np.random.randint(0, 100, N)

    # Sequential Execution
    start = time.time()
    C_seq = sequential_add(A, B)
    print("\nSequential Output:", C_seq)
    print("Sequential Time:", time.time() - start)

    # Parallel Execution
    start = time.time()
    C_par = parallel_add(A, B)
    print("\nParallel Output:", C_par)
    print("Parallel Time:", time.time() - start)

    # Verification
    print("\nResults same:", np.all(np.array(C_seq) == np.array(C_par)))
