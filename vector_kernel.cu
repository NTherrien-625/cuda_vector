#include <cuda_runtime.h>
#include "vector_kernel.cuh"

__global__ void init_Vectorf(float* v, unsigned int size) {
    // What thread are we on?
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Put a 0 in there
    if (i < size) v[i] = 0;
    
    // Get the hell out
    return;
}

__global__ void copy_Vectorf(float* lhs, float* rhs, unsigned int size) {
    // What thread are we on?
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy the value over
    if (i < size) lhs[i] = rhs[i];

    // Get the hell out
    return;
}
