#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << cv::getBuildInformation() << std::endl;
    return 0;
}
