/****************************************************************************80

    Image Processing using OpenCV and CUDA on GPUs

    Uses: 
    
    MIT License

    Copyright (c) 2024 Katalin Mecseki

******************************************************************************/


#include <opencv2/opencv.hpp>
#include <iostream>


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

width = inputImage.cols;
height = inputImage.rows;

std::cout << "Image loaded successfully, width: " << width << ", height: " << height << std::endl;






}