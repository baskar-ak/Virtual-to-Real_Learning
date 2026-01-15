/*

  Program to overlay virtual objects on to Aruco markers in the scene
  - Read a scene with marker
  - Detect markers in the scene
  - Read virtual object image
  - Overlay the virtual object onto the marker within the scene

*/

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

int main() {
    // Load the image containing the ArUco marker
    cv::Mat inputImage = cv::imread("Dataset/crossing/marker_scene/d15.png", cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not load input image." << std::endl;
        return -1;
    }

    // Detect ArUco markers
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);
    detector.detectMarkers(inputImage, markerCorners, markerIds, rejectedCandidates);

    // Check if markers are detected
    if (markerIds.empty()) {
        std::cerr << "Error: No ArUco markers detected." << std::endl;
        return -1;
    }

    // Load the virtual image to overlay
    cv::Mat overlay = cv::imread("Traffic Objects/duck/duck.png", cv::IMREAD_COLOR);
    if (overlay.empty()) {
        std::cerr << "Error: Could not load overlay image." << std::endl;
        return -1;
    }

    // Overlay the virtual object on each detected ArUco marker
    for (size_t i = 0; i < markerIds.size(); ++i) {
        // Get the bounding box of the detected marker
        cv::Rect markerRect = cv::boundingRect(markerCorners[i]);

        // Resize the overlay image to match the size of the detected marker
        cv::resize(overlay, overlay, markerRect.size());

        // Overlay the resized image on the marker
        overlay.copyTo(inputImage(markerRect));
    }

    // Display the result
    cv::namedWindow("Result", cv::WINDOW_NORMAL);
    cv::imshow("Result", inputImage);
    cv::imwrite("Dataset/duck/duck_014.png", inputImage); // Save the image
    cv::waitKey(0);

    return 0;

}


