#include <iostream>
#include <omp.h>

using namespace std;

void sequentialBubbleSort(int arr[], int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

void parallelBubbleSort(int arr[], int n)
{
    for (int i = 0; i < n; i++)
    {
        // Even phase
#pragma omp parallel for
        for (int j = 0; j < n - 1; j += 2)
        {
            if (arr[j] > arr[j + 1])
            {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }

        // Odd phase
#pragma omp parallel for
        for (int j = 1; j < n - 1; j += 2)
        {
            if (arr[j] > arr[j + 1])
            {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
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
        arr2[i] = arr[i]; // copy for parallel
    }

    double start, end;

    // Sequential timing
    start = omp_get_wtime();

    sequentialBubbleSort(arr, n);

    end = omp_get_wtime();

    cout << "\nSequential Sorted Array:\n";
    printArray(arr, n);

    cout << "Sequential Time: " << end - start << " seconds\n";

    // Parallel timing
    start = omp_get_wtime();

    parallelBubbleSort(arr2, n);

    end = omp_get_wtime();

    cout << "\nParallel Sorted Array:\n";
    printArray(arr2, n);

    cout << "Parallel Time: " << end - start << " seconds\n";

    return 0;
}
