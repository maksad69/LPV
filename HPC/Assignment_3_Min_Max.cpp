#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    int arr[n];

    cout << "Enter elements:\n";

    for (int i = 0; i < n; i++)
        cin >> arr[i];

    int sum = 0;
    int min = arr[0];
    int max = arr[0];

#pragma omp parallel for reduction(+:sum) reduction(min:min) reduction(max:max)
    for (int i = 0; i < n; i++)
    {
        sum += arr[i];

        if (arr[i] < min)
            min = arr[i];

        if (arr[i] > max)
            max = arr[i];
    }

    double avg = (double)sum / n;

    cout << "\nResults using Parallel Reduction:\n";
    cout << "Sum = " << sum << endl;
    cout << "Minimum = " << min << endl;
    cout << "Maximum = " << max << endl;
    cout << "Average = " << avg << endl;

    return 0;
}
