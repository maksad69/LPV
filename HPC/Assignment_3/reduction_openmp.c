#include <stdio.h>
#include <omp.h>

#define MAX 1000

int main() {
    int n, arr[MAX];

    printf("Enter number of elements: ");
    scanf("%d", &n);

    printf("Enter elements:\n");
    for (int i = 0; i < n; i++)
        scanf("%d", &arr[i]);

    int sum = 0;
    int min = arr[0];
    int max = arr[0];

    double start, end;

    start = omp_get_wtime();

    // SUM
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }

    // MIN
    #pragma omp parallel for reduction(min:min)
    for (int i = 0; i < n; i++) {
        if (arr[i] < min)
            min = arr[i];
    }

    // MAX
    #pragma omp parallel for reduction(max:max)
    for (int i = 0; i < n; i++) {
        if (arr[i] > max)
            max = arr[i];
    }

    end = omp_get_wtime();

    double avg = (double)sum / n;

    printf("\nSum = %d", sum);
    printf("\nMinimum = %d", min);
    printf("\nMaximum = %d", max);
    printf("\nAverage = %.2f", avg);

    printf("\nExecution Time = %f seconds\n", end - start);

    return 0;
}