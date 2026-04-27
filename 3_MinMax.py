import random
from multiprocessing import Pool, cpu_count

# ---------- Functions ----------
def chunk_sum(data):
    return sum(data)

def chunk_min(data):
    return min(data)

def chunk_max(data):
    return max(data)

# ---------- Parallel Reduction ----------
def parallel_reduce(arr, func):
    n = len(arr)
    num_workers = cpu_count()

    # Handle small input safely
    chunk_size = max(1, n // num_workers)

    # Divide into chunks
    chunks = [arr[i:i + chunk_size] for i in range(0, n, chunk_size)]

    with Pool(num_workers) as p:
        results = p.map(func, chunks)

    return results

# ---------- Main ----------
if __name__ == "__main__":

    choice = input("Enter '1' for manual input or '2' for random data: ")

    if choice == '1':
        arr = list(map(int, input("Enter numbers: ").split()))
    else:
        n = int(input("Enter size of array: "))
        arr = [random.randint(1, 1000) for _ in range(n)]

    n = len(arr)

    # Parallel Sum
    partial_sums = parallel_reduce(arr, chunk_sum)
    total_sum = sum(partial_sums)

    # Parallel Min
    partial_mins = parallel_reduce(arr, chunk_min)
    minimum = min(partial_mins)

    # Parallel Max
    partial_maxs = parallel_reduce(arr, chunk_max)
    maximum = max(partial_maxs)

    # Average
    average = total_sum / n

    # ---------- Output ----------
    print("\nArray:", arr)
    print("\nTotal Sum:", total_sum)
    print("Minimum:", minimum)
    print("Maximum:", maximum)
    print("Average:", average)
