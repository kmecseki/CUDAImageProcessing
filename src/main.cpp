/****************************************************************************80

    Image Processing using OpenCV and CUDA on GPUs

    Uses: 
    
    MIT License

    Copyright (c) 2024 Katalin Mecseki

******************************************************************************/


#include <opencv2/opencv.hpp>
#include <iostream>
#include "image_processing.h"
#include "utils.cpp"

int main(int argc, char* argv[]) {

if (argc<3) {
    std::cerr << " Usage: ./main <input> <output> [kernel_size] [sigma]" << std::endl;
    return -1;
}

std::string input = argv[1];
std::string output = argv[2];

int kernel_size = 5;
float sigma = 1.0f;

if (argc > 3) {
    kernel_size = std::stoi(argv[3]);
}
if (argc > 4) {
    sigma = std::stof(argv[4]);
}

cv::Mat inputImage = cv::imread(input, cv::IMREAD_COLOR);
if (inputImage.empty()) {
    std::cerr << "Image loading failed, file: " << input << std::endl;
    return -1;
}

int width = inputImage.cols;
int height = inputImage.rows;

std::cout << "Image loaded, width: " << width << ", height: " << height << std::endl;

/* Host memory pointers */

unsigned char *h_rgb = inputImage.data;


/* Allocate device memory */

unsigned char *d_rgb, *d_gray;
cudaCheckError(cudaMalloc((void**) &d_rgb, width * height * 3 * sizeof(unsigned char)));
cudaCheckError(cudaMalloc((void**) &d_gray, width * height * sizeof(unsigned char)));


/* Copy image to device */

cudaCheckError(cudaMemcpy(d_rgb, h_rgb, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));


/* Function that will call RGB -> Grayscale kernel */

cudaStream_t stream1;
cudaCheckError(cudaStreamCreate(&stream1));
std::cout << "Performing grayscale transformation" << std::endl;
RGBToGrayScale(d_rgb, d_gray, width, height, stream1);


/* Remove noise using gaussian blur */

cudaStream_t stream2;
cudaCheckError(cudaStreamCreate(&stream2));
std::cout << "Now Gaussian blur .. " << std::endl;
//float* d_kernel;
//float* h_kernel = GenKernel(kernel_size, sigma);
//cudaCheckError(cudaMalloc((void**) &d_kernel, kernel_size * kernel_size * sizeof(float)));
//cudaCheckError(cudaMemCpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float)), cudaMemcpyHostToDevice);
GaussianBlur(d_gray, d_blur, width, height, kernel_size, stream2);


cudaCheckError(cudaStreamSynchronize(stream1));

unsigned char *h_gray = new unsigned char[width * height];
cudaCheckError(cudaMemcpy(h_gray, d_gray, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost));

cv::Mat endresult(height, width, CV_8UC1, h_gray);
cv::imwrite(output, endresult);


delete h_gray;


}