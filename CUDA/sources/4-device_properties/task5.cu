#include <stdio.h>

#define NX 200
#define NY 100

__global__ void saxpy2D(float scalar, float * x, float * y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if ( row < NX && col < NY ) // Make sure we don't do more work than we have data!
        y[row*NY+col] = scalar * x[row*NY+col] + y[row*NY+col];
}

int main()
{
    float *x, *y;
    float maxError = 0;

    int size = NX * NY * sizeof (float); // The total number of bytes per vector

    cudaError_t ierrAsync;
    cudaError_t ierrSync;

    // Allocate memory
    cudaMallocManaged(&x, size);
    cudaMallocManaged(&y, size);

    // Initialize memory
    for( int i = 0; i < NX*NY; ++i )
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    cudaError_t ierr;
    
    ierr = cudaGetDeviceProperties(&prop, 0);
    if (ierr != cudaSuccess) { printf("Device property error: %s\n", cudaGetErrorString(ierr)); }
    printf("========== DEVICE PROPERTIES ==========\n");
    printf("Device number: %d\n", 0);
    printf("  Device name: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads in X-dimension of block: %d\n", prop.maxThreadsDim[0]);
    printf("  Max threads in Y-dimension of block: %d\n", prop.maxThreadsDim[1]);
    printf("  Max threads in Z-dimension of block: %d\n\n", prop.maxThreadsDim[2]);

    dim3 threads_per_block (1024,16,1);
    dim3 number_of_blocks ((NX/threads_per_block.x)+1,
                           (NY/threads_per_block.y)+1,
                           1);
     // Check total number of threads
    if (threads_per_block.x * threads_per_block.y * threads_per_block.z > prop.maxThreadsPerBlock) {
        printf("Max number of threads exceeded!\n"); goto cleanup;
    }
    // Check number of threads_per_block
    if (prop.maxThreadsDim[0] < threads_per_block.x && prop.maxThreadsDim[1] < threads_per_block.y) { printf("Block x- or y- sizes exceeded device limits!\n"); goto cleanup; }

    saxpy2D <<< number_of_blocks, threads_per_block >>> ( 2.0f, x, y );

    ierrSync = cudaGetLastError();
    ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
    if (ierrSync != cudaSuccess) { printf("Sync error: %s\n", cudaGetErrorString(ierrSync)); }
    if (ierrAsync != cudaSuccess) { printf("Async error: %s\n", cudaGetErrorString(ierrAsync)); }

    // Print out our Max Error
    for( int i = 0; i < NX*NY; ++i )
        if (abs(4-y[i]) > maxError) { maxError = abs(4-y[i]); }
    printf("Max Error: %.5f", maxError);

cleanup:
    // Free all our allocated memory
    cudaFree( x ); cudaFree( y );
}