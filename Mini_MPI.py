from mpi4py import MPI
import random
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------- Sequential Quicksort ----------
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# ---------- Main ----------
if __name__ == "__main__":

    if rank == 0:
        n = int(input("Enter number of elements: "))
        arr = [random.randint(1, 10000) for _ in range(n)]

        start_seq = time.time()
        sorted_seq = quicksort(arr.copy())
        end_seq = time.time()

        print("\nSequential Time:", end_seq - start_seq)
    else:
        arr = None

    # Scatter data
    arr = comm.bcast(arr, root=0)
    chunk = arr[rank::size]   # divide work

    # Parallel sorting
    start_par = MPI.Wtime()
    local_sorted = quicksort(chunk)

    gathered = comm.gather(local_sorted, root=0)

    if rank == 0:
        # Merge all sorted chunks
        final = []
        for part in gathered:
            final.extend(part)

        final_sorted = quicksort(final)
        end_par = MPI.Wtime()

        print("Parallel Time:", end_par - start_par)

        # Metrics
        speedup = (end_seq - start_seq) / (end_par - start_par)
        efficiency = speedup / size

        print("\nSpeedup:", speedup)
        print("Efficiency:", efficiency)

        print("Correct:", sorted_seq == final_sorted)
