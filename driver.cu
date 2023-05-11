#include <iostream>

#include "vector.cu"

int main() {
    Vectorf<10000>* gpu_vector = new Vectorf<10000>(true);
    delete(gpu_vector);
       
    return 0;
}
