#include <stdio.h>

__global__ void hello()
{
    printf("Hello from Thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main()
{
    hello<<<2, 64>>>();
    cudaDeviceSynchronize(); // The above call is asynchronous, wait until it
                             // finishes before exiting the program!

    return 0;
}