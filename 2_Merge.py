import random
import time
from multiprocessing import Pool

# ---------- Merge Function ----------
def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result

# ---------- Sequential Merge Sort ----------
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

# ---------- Parallel Merge Sort ----------
def parallel_merge_sort(arr):
    if len(arr) <= 1:
        return arr

    # Threshold to reduce overhead
    if len(arr) < 1000:
        return merge_sort(arr)

    mid = len(arr) // 2

    with Pool(2) as p:
        left, right = p.map(parallel_merge_sort, [arr[:mid], arr[mid:]])

    return merge(left, right)

# ---------- Main ----------
if __name__ == "__main__":

    choice = input("Enter '1' for manual input or '2' for random array: ")

    if choice == '1':
        arr = list(map(int, input("Enter elements: ").split()))
    else:
        n = int(input("Enter size of array: "))
        arr = [random.randint(1, 100000) for _ in range(n)]

    arr_seq = arr.copy()
    arr_par = arr.copy()

    # Sequential Execution
    start = time.time()
    sorted_seq = merge_sort(arr_seq)
    print("\nSequential Output:", sorted_seq)
    print("Sequential Time:", time.time() - start)

    # Parallel Execution
    start = time.time()
    sorted_par = parallel_merge_sort(arr_par)
    print("\nParallel Output:", sorted_par)
    print("Parallel Time:", time.time() - start)

    # Verification
    print("\nOutputs are same:", sorted_seq == sorted_par)
