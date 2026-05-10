#include <stdio.h>
#include <omp.h>

#define MAX 10

int graph[MAX][MAX];
int visited[MAX];
int queue[MAX], front = 0, rear = 0;

// ---------- BFS FUNCTIONS ----------
void enqueue(int v) {      // add element in queue
    queue[rear++] = v;     // here rear is queue index
}

int dequeue() {            // remove element from queue
    return queue[front++];
}

void parallel_bfs(int start, int n) {    // start = starting node,  n = number of nodes
    front = rear = 0;                    // Queue is empty initially. here front and rear represents index of queue.

    for (int i = 0; i < n; i++){    // Mark all nodes as: “Not visited”
        visited[i] = 0;
    }

    enqueue(start);     // Add starting node to queue
    visited[start] = 1; // Mark it as visited

    printf("BFS Traversal: ");

    while (front < rear) {  // Loop runs until queue is empty
        int node = dequeue();  // Remove node
        printf("%d ", node);   // Print it

        #pragma omp parallel for  // OpenMP creates multiple threads  and  Each thread checks neighbors of node.   “Check all neighbors in parallel”
        for (int i = 0; i < n; i++) {    // this loop runs in parallel 
            if (graph[node][i] && !visited[i]) {   // Condition means: There is an edge (connection). Node is not visited
                #pragma omp critical  // Only ONE thread enters at a time. Why? To avoid errors when multiple threads modify queue
                {
                    if (!visited[i]) {  
                        enqueue(i);       // Add new node to queue.
                        visited[i] = 1;  // Mark it visited
                    }
                }
            }
        }
    }
}

// ---------- DFS FUNCTION ----------
void parallel_dfs(int node, int n) {  // node = current node, n = total number of nodes
    visited[node] = 1;     // This node is now visited
    printf("%d ", node);

    for (int i = 0; i < n; i++) {
        if (graph[node][i] && !visited[i]) {    // graph[node][i] → connection exists and node not visited
            #pragma omp task         // Give this small work to any free thread
            {  
                parallel_dfs(i, n);
            }
        }
    }
}

// ---------- MAIN ----------
int main() {
    int n, start, choice;

    printf("Enter number of vertices: ");  // User enters how many nodes are in graph
    scanf("%d", &n);

    printf("Enter adjacency matrix:\n");   // 1 → connection exists   0 → no connection
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    printf("Enter starting node: ");  // From which node traversal begins
    scanf("%d", &start);

    printf("\n1. BFS\n2. DFS\nEnter your choice: ");
    scanf("%d", &choice);

    // Reset visited array
    for (int i = 0; i < n; i++){  // This sets all nodes as not visited. Important because: We don’t want to visit same node again
        visited[i] = 0;
    }

    if (choice == 1) {
        parallel_bfs(start, n);
    } 
    else if (choice == 2) {
        printf("DFS Traversal: ");

        #pragma omp parallel     // Creates multiple threads
        {
            #pragma omp single  // Only one thread starts the DFS. Why? To avoid multiple threads starting same recursion
            {
                parallel_dfs(start, n);
            }
        }
    } 
    else {
        printf("Invalid choice!");
    }

    return 0;
}