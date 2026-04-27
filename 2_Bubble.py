import random
import time
from multiprocessing import Pool, cpu_count

# ---------- Sequential Bubble Sort ----------
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# ---------- Parallel Bubble Sort (Odd-Even Transposition) ----------
def compare_swap(pair):
    a, b = pair
    return (min(a, b), max(a, b))

def parallel_bubble_sort(arr):
    n = len(arr)

    for i in range(n):
        pairs = []

        # Even phase / Odd phase
        start = 0 if i % 2 == 0 else 1

        for j in range(start, n - 1, 2):
            pairs.append((arr[j], arr[j + 1]))

        with Pool(cpu_count()) as p:
            results = p.map(compare_swap, pairs)

        idx = 0
        for j in range(start, n - 1, 2):
            arr[j], arr[j + 1] = results[idx]
            idx += 1

    return arr

# ---------- Main ----------
if __name__ == "__main__":

    choice = input("Enter '1' for manual input or '2' for random array: ")

    if choice == '1':
        arr = list(map(int, input("Enter elements: ").split()))
    else:
        n = int(input("Enter size of array: "))
        arr = [random.randint(1, 10000) for _ in range(n)]

    arr_seq = arr.copy()
    arr_par = arr.copy()

    # Sequential Bubble Sort
    start = time.time()
    bubble_sort(arr_seq)
    print("\nSequential Output:", arr_seq)
    print("Sequential Time:", time.time() - start)

    # Parallel Bubble Sort
    start = time.time()
    parallel_bubble_sort(arr_par)
    print("\nParallel Output:", arr_par)
    print("Parallel Time:", time.time() - start)

    # Verification
    print("\nOutputs are same:", arr_seq == arr_par)
