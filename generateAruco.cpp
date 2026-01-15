/*
  Program to generate ArUco markers
*/

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

int main() {

    cv::Mat markerImage;
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50); // 4x4 Dictionary
    cv::aruco::generateImageMarker(dictionary, 23, 200, markerImage, 1); // Aruco marker with ID 23
    cv::imwrite("marker23.png", markerImage);

    std::cout << "Marker created!\n";

    return 0;
}
