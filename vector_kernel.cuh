#ifndef VECTOR_KERNEL_H
#define VECTOR_KERNEL_H

#include <cuda_runtime.h>

/*
           Name : init_Vectorf(float* v)
 Pre-conditions : v is an array of floats allocated on a CUDA device
Post-conditions : v is populated with 0s
*/
__global__ void init_Vectorf(float* v, unsigned int size);

/*
           Name : copy_Vectorf(float* lhs, float* rhs)
 Pre-conditions : lhs and rhs are arrays of floats allocated on a CUDA device
Post-conditions : lhs is populated with the contents of rhs
*/
__global__ void copy_Vectorf(float* lhs, float* rhs, unsigned int size);

#endif
