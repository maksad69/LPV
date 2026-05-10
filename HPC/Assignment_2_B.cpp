#include <iostream>
#include <omp.h>

using namespace std;

void merge(int arr[], int l, int m, int r)
{
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[n1], R[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];

    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];

    while (j < n2)
        arr[k++] = R[j++];
}

void sequentialMergeSort(int arr[], int l, int r)
{
    if (l < r)
    {
        int m = (l + r) / 2;

        sequentialMergeSort(arr, l, m);
        sequentialMergeSort(arr, m + 1, r);

        merge(arr, l, m, r);
    }
}

void parallelMergeSort(int arr[], int l, int r)
{
    if (l < r)
    {
        int m = (l + r) / 2;

#pragma omp parallel sections
        {
#pragma omp section
            parallelMergeSort(arr, l, m);

#pragma omp section
            parallelMergeSort(arr, m + 1, r);
        }

        merge(arr, l, m, r);
    }
}

void printArray(int arr[], int n)
{
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";

    cout << endl;
}

int main()
{
    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    int arr[n], arr2[n];

    cout << "Enter elements:\n";

    for (int i = 0; i < n; i++)
    {
        cin >> arr[i];
        arr2[i] = arr[i];
    }

    double start, end;

    // Sequential Merge Sort
    start = omp_get_wtime();

    sequentialMergeSort(arr, 0, n - 1);

    end = omp_get_wtime();

    cout << "\nSequential Sorted Array:\n";
    printArray(arr, n);

    cout << "Sequential Time: " << end - start << " seconds\n";

    // Parallel Merge Sort
    start = omp_get_wtime();

    parallelMergeSort(arr2, 0, n - 1);

    end = omp_get_wtime();

    cout << "\nParallel Sorted Array:\n";
    printArray(arr2, n);

    cout << "Parallel Time: " << end - start << " seconds\n";

    return 0;
}
