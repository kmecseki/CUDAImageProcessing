#include <iostream>
#include <cuda_runtime.h>

inline void gpuCheckError(cudaError_t caller, const char *file, int line, bool abort) {
    if (caller!=cudaSuccess) {
        std::cerr << "CUDA problem at " << file << ", line: " <<  line << std::endl;
        if (abort) {
            exit(-1);
        }
    }
}

