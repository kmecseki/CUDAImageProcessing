/****************************************************************************80

    Image Processing using OpenCV and CUDA on GPUs

    Uses: 
    
    MIT License

    Copyright (c) 2024 Katalin Mecseki

******************************************************************************/


#include <opencv2/opencv.hpp>
#include <iostream>
#include "image_processing.h"
#include "utils.h"

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


unsigned char *h_rgb = inputImage.data;
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
float *h_kernel = new float(kernel_size * kernel_size);
h_kernel = genKernel(kernel_size, sigma);
float* d_kernel;
cudaCheckError(cudaMalloc((void**) &d_kernel, kernel_size * kernel_size * sizeof(float)));
cudaCheckError(cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));
unsigned char *d_blur;
cudaCheckError(cudaMalloc((void**) &d_blur, width * height * sizeof(unsigned char)));
GaussianBlur(d_gray, d_blur, d_kernel, width, height, kernel_size, stream2, sigma);


/* Kernel check */
//float *temp = new float[kernel_size * kernel_size];
//cudaCheckError(cudaMemcpy(temp, d_kernel, sizeof(float) * kernel_size * kernel_size, cudaMemcpyDeviceToHost));
//cv::Mat endresult(kernel_size, kernel_size, CV_8UC1, temp);
//cv::imwrite(output, endresult);

//for (int i=0; i<kernel_size*kernel_size; ++i) {
//    if ((i+1)%kernel_size == 0) {
//        std::cout << temp[i] << std::endl;
//    } else {
//        std::cout << temp[i] << " ";
//    }
//}
//std::cout << std::endl;

cudaStream_t stream3;
cudaCheckError(cudaStreamCreate(&stream3));
unsigned char *d_edge;
cudaCheckError(cudaMalloc((void**) &d_edge, width * height * sizeof(unsigned char)));
sobelEdgeDetection(d_blur, d_edge, width, height, stream3);


cudaCheckError(cudaStreamSynchronize(stream1));
cudaCheckError(cudaStreamSynchronize(stream2));
cudaCheckError(cudaStreamSynchronize(stream3));


unsigned char *out = new unsigned char[width * height];
cudaCheckError(cudaMemcpy(out, d_edge, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost));

cv::Mat endresult(height, width, CV_8UC1, out);
cv::imwrite(output, endresult);


delete[] out;
delete[] h_kernel;
cudaCheckError(cudaFree(d_rgb));
cudaCheckError(cudaFree(d_gray));
cudaCheckError(cudaFree(d_blur));
cudaCheckError(cudaFree(d_edge));
cudaCheckError(cudaFree(d_kernel));
cudaCheckError(cudaStreamDestroy(stream1));
cudaCheckError(cudaStreamDestroy(stream2));
cudaCheckError(cudaStreamDestroy(stream3));

return 0;

}