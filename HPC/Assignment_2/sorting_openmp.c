#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX 1000

// ---------- PRINT FUNCTION ----------
void print_array(int arr[], int n) {
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

// ---------- BUBBLE SORT ----------
void bubble_sort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

// Parallel Bubble Sort (Odd-Even Sort)
void parallel_bubble_sort(int arr[], int n) {
    for (int i = 0; i < n; i++) {   // this loop is pass which runs sequentially
        #pragma omp parallel for    // here threads are created
        for (int j = (i % 2); j < n-1; j += 2) {   
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

// ---------- MERGE SORT ----------
void merge(int arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;   // Size of left part
    int n2 = r - m;       // Size of right part

    int L[n1], R[n2];    // create temporary arrays L for left and r for right

    // Copy values from original array into L and R
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0; j = 0; k = l;  // i → left array index,  j → right array index,  k → main array index

    while (i < n1 && j < n2) {  // Compare elements from both arrays
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)      // If left side remains
        arr[k++] = L[i++];

    while (j < n2)      // If right side remains:
        arr[k++] = R[j++];
}

void merge_sort(int arr[], int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        merge_sort(arr, l, m);   // runs first
        merge_sort(arr, m+1, r); // runs after first finishes
        merge(arr, l, m, r);
    }
}

// Parallel Merge Sort
void parallel_merge_sort(int arr[], int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallel_merge_sort(arr, l, m);

            #pragma omp section
            parallel_merge_sort(arr, m+1, r);
        }

        merge(arr, l, m, r);
    }
}

// ---------- MAIN ----------
int main() {
    int n;
    int original[MAX], arr1[MAX], arr2[MAX];   // This creates 3 arrays

    printf("Enter number of elements: ");
    scanf("%d", &n);

    printf("Enter elements:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &original[i]);
    }

    double start, end;

    // Sequential Bubble
    for (int i = 0; i < n; i++){
        arr1[i] = original[i];
    }

    start = omp_get_wtime();   // Start measuring time from here (it take systems current time)
    bubble_sort(arr1, n);
    end = omp_get_wtime();     // Stop measuring time here
    printf("\nSequential Bubble Sort Time: %f\n", end - start);
    printf("Sorted (Sequential Bubble): ");
    print_array(arr1, n);

    // Parallel Bubble
    for (int i = 0; i < n; i++)
        arr2[i] = original[i];

    start = omp_get_wtime();
    parallel_bubble_sort(arr2, n);
    end = omp_get_wtime();
    printf("\nParallel Bubble Sort Time: %f\n", end - start);
    printf("Sorted (Parallel Bubble): ");
    print_array(arr2, n);

    // Sequential Merge
    for (int i = 0; i < n; i++)
        arr1[i] = original[i];

    start = omp_get_wtime();
    merge_sort(arr1, 0, n-1);
    end = omp_get_wtime();
    printf("\nSequential Merge Sort Time: %f\n", end - start);
    printf("Sorted (Sequential Merge): ");
    print_array(arr1, n);

    // Parallel Merge
    for (int i = 0; i < n; i++)
        arr2[i] = original[i];

    start = omp_get_wtime();
    parallel_merge_sort(arr2, 0, n-1);
    end = omp_get_wtime();
    printf("\nParallel Merge Sort Time: %f\n", end - start);
    printf("Sorted (Parallel Merge): ");
    print_array(arr2, n);

    return 0;
}