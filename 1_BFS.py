from multiprocessing import Pool, Manager

# Function to process each node in parallel
def process_node(args):
    node, visited, graph = args
    result = []
    for neighbor in graph[node]:
        if neighbor not in visited:
            result.append(neighbor)
    return result

# Parallel BFS function
def parallel_bfs(start, graph):
    manager = Manager()
    visited = manager.list([start])
    queue = [start]

    print("\nBFS Traversal:")

    while queue:
        print(queue, end=" ")

        with Pool() as pool:
            results = pool.map(process_node, [(node, visited, graph) for node in queue])

        new_nodes = []
        for res in results:
            for node in res:
                if node not in visited:
                    visited.append(node)
                    new_nodes.append(node)

        queue = new_nodes


if __name__ == "__main__":
    graph = {}

    n = int(input("Enter number of nodes: "))

    print("Enter adjacency list for each node:")
    for i in range(n):
        neighbors = list(map(int, input(f"Neighbors of node {i}: ").split()))
        graph[i] = neighbors

    start = int(input("Enter starting node: "))

    parallel_bfs(start, graph)
