#include <stdio.h>
#include <math.h>
#include <iostream>
// CUDA
#include <cuda.h>
#include "util.hpp"

// Simple define to index into a 1D array from 2D space
#define I2D(num, c, r) ((r)*(num)+(c))

__global__
void step_kernel_mod(int ni, int nj, float fact, float* temp_in, float* temp_out) 
{ 
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;  
  
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  
  // loop over all points in domain (except boundary) 
  if (j > 0 && i > 0 && j < nj-1 && i < ni-1) {
    // find indices into linear memory 
    // for central point and neighbours 
    i00 = I2D(ni, i, j); 
    im10 = I2D(ni, i-1, j); 
    ip10 = I2D(ni, i+1, j); 
    i0m1 = I2D(ni, i, j-1); 
    i0p1 = I2D(ni, i, j+1); 

    // evaluate derivatives 
    d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10]; 
    d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1]; 

    // update temperatures 
    temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2); 
  }
}

void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out) 
{ 
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;  
  
  
  // loop over all points in domain (except boundary) 
  for ( int j=1; j < nj-1; j++ ) { 
    for ( int i=1; i < ni-1; i++ ) { 
      // find indices into linear memory 
      // for central point and neighbours 
      i00 = I2D(ni, i, j); 
      im10 = I2D(ni, i-1, j); 
      ip10 = I2D(ni, i+1, j); 
      i0m1 = I2D(ni, i, j-1); 
      i0p1 = I2D(ni, i, j+1); 
            
      // evaluate derivatives 
      d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10]; 
      d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1]; 
            
      // update temperatures 
      temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2); 
    } 
  } 
} 

int main()
{
  int istep;
  int nstep = 200; // number of time steps
  
  // Specify our 2D dimensions
  const int ni = 2000;
  const int nj = 1000;
  float tfac = 8.418e-5; // thermal diffusivity of silver
  
  float *temp1_ref, *temp2_ref, *temp1, *temp2, *temp_tmp;
  
  const size_t size = ni * nj;
  
  temp1_ref = malloc_host<float>(size);
  temp2_ref = malloc_host<float>(size);
  // Here we use Unified Memory
  temp1 = malloc_managed<float>(size);
  temp2 = malloc_managed<float>(size);
  
  // Initialize with random data
  for( int i = 0; i < ni*nj; ++i) {
    temp1_ref[i] = temp2_ref[i] = temp1[i] = temp2[i] = (float)rand()/(float)(RAND_MAX/100.0f);
  }
  
  // Execute the CPU-only reference version
  auto start_cpu = get_time();
  for (istep=0; istep < nstep; istep++) { 
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref); 
       
    // swap the temperature pointers 
    temp_tmp = temp1_ref; 
    temp1_ref = temp2_ref; 
    temp2_ref= temp_tmp; 
  }
  auto total_cpu = get_time() - start_cpu;
  std::cerr << "CPU time: " << total_cpu << "s\n";

  dim3 tblocks(32, 16, 1);
  dim3 grid((nj/tblocks.x)+1, (ni/tblocks.y)+1, 1);
  
  // Execute the modified version using same data
  auto start_gpu = get_time();
  for (istep=0; istep < nstep; istep++) { 
    step_kernel_mod<<< grid, tblocks >>>(ni, nj, tfac, temp1, temp2); 
    
    // Check errors
    cuda_check_last_kernel("step_kernel_mod");
    cuda_check_status(cudaDeviceSynchronize());
       
    // swap the temperature pointers 
    temp_tmp = temp1; 
    temp1 = temp2; 
    temp2= temp_tmp; 
  }
  
  auto total_gpu = get_time() - start_gpu;
  std::cerr << "GPU time: " << total_gpu << "s\n";
  
  float maxError = 0;
  // Output should always be stored in the temp1 and temp1_ref at this point
  for( int i = 0; i < ni*nj; ++i ) {
    if (abs(temp1[i]-temp1_ref[i]) > maxError) { maxError = abs(temp1[i]-temp1_ref[i]); }
  }
  
  // Check and see if our maxError is greater than an error bound
  if (maxError > 0.0005f)
  	printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
  else
  	printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);
  
  free( temp1_ref );
  free( temp2_ref );
  cudaFree( temp1 );
  cudaFree( temp2 );
    
  return 0;
}