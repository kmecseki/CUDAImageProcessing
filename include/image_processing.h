#ifndef CUDA_IMAGE_PROCESSING_H
#define CUDA_IMAGE_PROCESSING_H

#include <cuda_runtime.h>

#define cudaCheckError(ans) {gpuCheckError((ans), __FILE__, __LINE__);}

inline void gpuCheckError(cudaError caller, const char *file, int line, bool abort=true);

/* GPU kernel functions */



#endif

