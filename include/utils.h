#pragma once
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include "image_processing.h"

inline void gpuCheckError(cudaError_t caller, const char *file, int line, bool abort) {
    if (caller!=cudaSuccess) {
        std::cerr << "CUDA problem at " << file << ", line: " <<  line << std::endl;
        if (abort) {
            exit(-1);
        }
    }
}

float* genKernel(int kernel_size, float sigma) {
    float* kernel = new float[kernel_size * kernel_size];
    int half = kernel_size / 2;
    float sum = 0.0f;
    float s = 2.0f * sigma * sigma;
    float s2 = 3.1415926 * s;

    for(int y = -half; y <= half; y++) {
        for(int x = -half; x <= half; x++) {
            int idx = (y + half) * kernel_size + (x + half);
            kernel[idx] = expf(-(x * x + y * y) / s) / s2;
            sum += kernel[idx];
        }
    }

    for(int i=0; i<kernel_size*kernel_size; i++) {
        kernel[i] /= sum;
    }
    return kernel;
}

#endif