#include <stdio.h>

#define N 2048 * 2048 // Number of elements in each vector

// TODO: kernel keyword
void saxpy(int * a, int * b, int * c)
{
    // TODO: Determine our unique global thread ID, so we know which element to process
    
    if ( tid < N ) // Make sure we don't do more work than we have data!
        // TODO: write the kernel
}

int main()
{
    int *a, *b, *c;

    int size = N * sizeof (int); // The total number of bytes per vector

    // Allocate memory
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    // Initialize memory
    for( int i = 0; i < N; ++i )
    {
        a[i] = 2;
        b[i] = 1;
        c[i] = 0;
    }

    int threads_per_block = 128;
    int number_of_blocks = (N / threads_per_block) + 1;

    // TODO: launch the kernel

    // TODO: Wait for the GPU to finish

    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");

    // Free all our allocated memory
    cudaFree( a ); cudaFree( b ); cudaFree( c );
}
