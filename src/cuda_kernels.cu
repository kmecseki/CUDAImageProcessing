
__device__ void genKernelKernel(float *kernel, int kernel_size, float sigma) {
    int idxx = blockIdx.x * blockDim.x + threadIdx.x;
    int idxy = blockIdx.y * blockDim.y + threadIdx.y;

    float s = 2.0f * sigma * sigma;
    float s2 = 3.1415926 * s;
    float sum = 0.0f;
    int half = kernel_size/2;

    if (idxx < kernel_size && idxy < kernel_size) {
        kernel[idxy * kernel_size + idxx] = expf(-(idxx-half * idxx-half + idxy-half * idxy-half) / s) / s2;
        sum += kernel[idxy * kernel_size + idxx];
    }

    if (idxx < kernel_size && idxy < kernel_size) {
        kernel[idxy * kernel_size + idxx] /= sum;
    }
}


__global__ void RGBToGrayscaleKernel(const unsigned char* rgb, unsigned char* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        float r = rgb[idx];
        float g = rgb[idx + 1];
        float b = rgb[idx + 2];
        /* NTSC formula */
        gray[y * width + x] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}


void RGBToGrayScale(const unsigned char* d_rgb, unsigned char* d_gray, int width, int height, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1)/block.x, (height + block.y -1 )/block.y);

    RGBToGrayscaleKernel<<<grid, block, 0, stream>>>(d_rgb, d_gray, width, height);

}


__global__ void GaussianBlurKernel(const unsigned char* d_gray, unsigned char *d_blur, int width, int height, int kernel_size) {
    int a = 1;
}


void GaussianBlur(const unsigned char *d_gray, unsigned char *d_blur, int width, int height, int kernel_size, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1 )/block.x, (height + block.y -1)/block.y);
    /* use shared memory for the image so all threads can work on it fast */
    /* TODO: add check if it fits with current hardware */
    int sharedmem = (block.x + kernel_size) * (block.y + kernel_size);
    GaussianBlurKernel<<<grid, block, sharedmem, stream>>>(d_gray, d_blur, width, height, kernel_size);
    
    
    
    dim3 block;
    if (kernel_size > 8){
        block = dim3(8, 8);
    } 
    else {
        block = dim3(kernel_size, kernel_size);
    }
    dim3 grid()

}
