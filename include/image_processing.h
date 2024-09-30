#pragma once

#ifndef CUDA_IMAGE_PROCESSING_H
#define CUDA_IMAGE_PROCESSING_H

#include <cuda_runtime.h>

#define cudaCheckError(ans) {gpuCheckError((ans), __FILE__, __LINE__);}

inline void gpuCheckError(cudaError_t caller, const char *file, int line, bool abort=true);


/* GPU kernel functions */

__global__ void RGBToGrayscaleKernel(const unsigned char* rgb, unsigned char* gray, int width, int height);
__device__ void genKernelKernel(float *kernel, int kernel_size, float sigma);
__global__ void GaussianBlurKernel(const unsigned char* input, unsigned char *output, float *kernel, int width, int height, int kernel_size, cudaStream_t stream, float sigma);

/* Host functions */

void RGBToGrayScale(const unsigned char *input, unsigned char *output, int width, int height, cudaStream_t stream);
void GaussianBlur(const unsigned char *input, unsigned char *output, float *kernel, int width, int height, int kernel_size, cudaStream_t stream, float sigma);
float* genKernel(int kernel_size, float sigma);


#endif

