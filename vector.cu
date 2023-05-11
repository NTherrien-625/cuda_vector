#ifndef VECTOR_CU
#define VECTOR_CU

#include <stdio.h>
#include "vector_kernel.cuh"

template <unsigned int T>
class Vectorf {
public:
    // Default constructor
    Vectorf();

    // Overloaded constructor
    Vectorf(bool gpu);

    // Destructor
    ~Vectorf();

    float* cpu_data;
    float* gpu_data;

private:
    bool use_gpu;
    unsigned int thread_usage;
    unsigned int block_usage;
};

template <unsigned int T>
Vectorf<T>::Vectorf() {
    // Allocate the data array
    cpu_data = (float*) malloc(T * sizeof(float));
    
    // Init the array
    for (unsigned int i = 0; i < T; ++i)
        cpu_data[i] = 0;

    // By default do not use the gpu
    use_gpu = false;
    gpu_data = nullptr;
}

template <unsigned int T>
Vectorf<T>::Vectorf(bool gpu) {
    // Use a gpu if it is available
    bool gpu_available;
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        gpu_available = false;
    }
    else {
        gpu_available = true;
    }
    if (gpu_available && gpu) {
        // Enable and allocate the vector
        use_gpu = true;
        cudaMalloc(&gpu_data, T * sizeof(float));

        // Find the thread and block usage of the device
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        unsigned int block_size = props.maxBlocksPerMultiProcessor;
        thread_usage = block_size;
        block_usage = ceil(T / block_size);

        // Init the vector
        init_Vectorf<<<block_usage, thread_usage>>>(gpu_data, T);
    }

    // Else use the default
    else {
        cpu_data = (float*) malloc(T * sizeof(float));

        for (unsigned int i = 0; i < T; ++i)
            cpu_data[i] = 0;

        use_gpu = false;
        gpu_data = nullptr;
    }
}

template <unsigned int T>
Vectorf<T>::~Vectorf() {
    // Free from the gpu
    if (use_gpu) {
        cudaFree(gpu_data);
        gpu_data = nullptr;
    }

    // Free from the cpu
    else {
        free(cpu_data);
        cpu_data = nullptr;
    }
}

#endif
