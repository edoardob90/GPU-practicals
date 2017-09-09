#include <stdio.h>

#define N  64

// TODO: kernel keyword
void matrixMulGPU( int * a, int * b, int * c )
{
    int val = 0;

    int row = // TODO: get row id of current thread
    int col = // TODO: get col id of current thread

    if (row < N && col < N)
    {
        for ( int k = 0; k < N; ++k )
            val += a[row * N + k] * b[k * N + col];
        c[row * N + col] = val;
    }
}

void matrixMulCPU( int * a, int * b, int * c )
{
    int val = 0;

    for( int row = 0; row < N; ++row )
        for( int col = 0; col < N; ++col )
        {
            val = 0;
            for ( int k = 0; k < N; ++k )
                val += a[row * N + k] * b[k * N + col];
            c[row * N + col] = val;
        }
}

int main()
{
    int *a, *b, *c_cpu, *c_gpu;

    int size = N * N * sizeof (int); // Number of bytes of an N x N matrix

    // Allocate memory
    cudaMallocManaged (&a, size);
    cudaMallocManaged (&b, size);
    cudaMallocManaged (&c_cpu, size);
    cudaMallocManaged (&c_gpu, size);

    // Initialize memory
    for( int row = 0; row < N; ++row )
        for( int col = 0; col < N; ++col )
        {
            a[row*N + col] = row;
            b[row*N + col] = col+2;
            c_cpu[row*N + col] = 0;
            c_gpu[row*N + col] = 0;
        }

    // TODO: define max number of threads per block
    // TODO: define number of blocks
    // Use CUDA-defined type 'dim3'

    matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu );

    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

    // Call the CPU version to check our work
    matrixMulCPU( a, b, c_cpu );

    // Compare the two answers to make sure they are equal
    bool error = false;
    for( int row = 0; row < N && !error; ++row )
        for( int col = 0; col < N && !error; ++col )
            if (c_cpu[row * N + col] != c_gpu[row * N + col])
            {
                printf("FOUND ERROR at c[%d][%d]\n", row, col);
                error = true;
                break;
            }
    if (!error)
        printf("Success!\n");

    // Free all our allocated memory
    cudaFree(a); cudaFree(b);
    cudaFree( c_cpu ); cudaFree( c_gpu );
}
