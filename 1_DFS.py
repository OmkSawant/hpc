from multiprocessing import Pool, Manager

# DFS utility function
def dfs_util(node, visited, graph):
    if node not in visited:
        visited.append(node)
        print(node, end=" ")

        for neighbor in graph[node]:
            dfs_util(neighbor, visited, graph)

# Parallel DFS function
def parallel_dfs(start, graph):
    manager = Manager()
    visited = manager.list()

    print("\nDFS Traversal:")

    # Note: Still mostly sequential due to DFS nature
    with Pool() as pool:
        pool.apply(dfs_util, args=(start, visited, graph))


if __name__ == "__main__":
    graph = {}

    n = int(input("Enter number of nodes: "))

    print("Enter adjacency list for each node:")
    for i in range(n):
        neighbors = list(map(int, input(f"Neighbors of node {i}: ").split()))
        graph[i] = neighbors

    start = int(input("Enter starting node: "))

    parallel_dfs(start, graph)
