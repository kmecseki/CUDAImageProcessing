
__device__ void genKernelKernel(float *kernel, int kernel_size, float sigma) {
    int idxx = blockIdx.x * blockDim.x + threadIdx.x;
    int idxy = blockIdx.y * blockDim.y + threadIdx.y;

    float s = 2.0f * sigma * sigma;
    float s2 = 3.1415926 * s;
    float sum = 0.0f;
    int half = kernel_size/2;

    if (idxx < kernel_size && idxy < kernel_size) {
        kernel[idxy * kernel_size + idxx] = expf(-((idxx-half) * (idxx-half) + (idxy-half) * (idxy-half)) / s) / s2;
        //sum += kernel[idxy * kernel_size + idxx];
    }
    
    //if (idxx < kernel_size && idxy < kernel_size) {
    //    kernel[idxy * kernel_size + idxx] /= sum;
    //}
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


__global__ void GaussianBlurKernel(const unsigned char* input, unsigned char *output, float *kernel, int width, int height, int kernel_size, float sigma) {

    //genKernelKernel(kernel, kernel_size, sigma);

    extern __shared__ unsigned char sharedmem[];
    int radius = kernel_size/2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedw = blockDim.x + 2 * radius;

    int localX = threadIdx.x + radius;
    int localY = threadIdx.y + radius;

    if(x < width && y < height) {
        sharedmem[localY * sharedw + localX] = input[y * width + x];
        
        if(threadIdx.x < radius && x >= radius)
            sharedmem[localY * sharedw + (localX - radius)] = input[y * width + (x - radius)];
        if(threadIdx.x >= blockDim.x - radius && x + radius < width)
            sharedmem[localY * sharedw + (localX + radius)] = input[y * width + (x + radius)];
        if(threadIdx.y < radius && y >= radius)
            sharedmem[(localY - radius) * sharedw + localX] = input[(y - radius) * width + x];
        if(threadIdx.y >= blockDim.y - radius && y + radius < height)
            sharedmem[(localY + radius) * sharedw + localX] = input[(y + radius) * width + x];
    }
    __syncthreads();

    if(x < width && y < height) {
        float sum = 0.0f;
        for(int ky = -radius; ky <= radius; ky++)
        {
            for(int kx = -radius; kx <= radius; kx++)
            {
                int imgX = localX + kx;
                int imgY = localY + ky;
                sum += kernel[(ky + radius) * kernel_size + (kx + radius)] * sharedmem[imgY * sharedw + imgX];
            }
        }
        output[y * width + x] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
    }
}


void GaussianBlur(const unsigned char *input, unsigned char *output, float *kernel, int width, int height, int kernel_size, cudaStream_t stream, float sigma) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1 )/block.x, (height + block.y -1)/block.y);
    /* use shared memory for the image so all threads can work on it fast */
    /* TODO: add check if it fits with current hardware */
    int sharedmem = (block.x + kernel_size) * (block.y + kernel_size);
    GaussianBlurKernel<<<grid, block, sharedmem, stream>>>(input, output, kernel, width, height, kernel_size, sigma);
    
}


__global__ void sobelEdgeDetectionKernel(const unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.x * blockDim.x + threadIdx.y;

    if (x>0 && x < (width - 1) && y>0 && y < (height - 1)) {
        int idx = y * width + x;
        int gx = -input[(y-1)*width + (x-1)] - 2 * input[(y)*width + (x-1)] - input[(y+1)*width + (x-1)] + input[(y-1)*width + (x+1)] + 2 * input[y*width + (x+1)] + input[(y+1)*width + (x+1)];
        
        int gy =  input[(y-1)*width + (x-1)] + 2 * input[(y-1)*width + (x)] + input[(y-1)*width + (x+1)] - input[(y+1)*width + (x-1)] - 2 * input[(y+1)*width + (x)] - input[(y+1)*width + (x+1)];
        //int gy =  input[(y+1)*width + (x-1)] + 2 * input[(y+1)*width + x] + input[(y+1)*width + (x+1)] - input[(y-1)*width + (x-1)] - 2 * input[(y-1)*width + x] - input[(y-1)*width + (x+1)];
        /* using abs. approx.: */
        int G = abs(gx) + abs(gy);
        output[idx] = G > 255 ? 255 : G;
    }
}


void sobelEdgeDetection(const unsigned char *input, unsigned char *output, int width, int height, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    sobelEdgeDetectionKernel<<<grid, block, 0, stream>>>(input, output, width, height);
}