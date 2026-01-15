/*

  Program to overlay virtual objects on to Aruco markers in the scene on live video stream
  - Read an image with marker
  - Read virtual object image
  - Overlay the virtual object onto the marker on a live video feed

*/

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

int main() {
    
    cv::VideoCapture cap(0); 

    // Check if the camera opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Failed to open camera." << std::endl;
        return -1;
    }

    // Create a window to display the output
    cv::namedWindow("Result", cv::WINDOW_NORMAL);

    // Load the image to overlay
    cv::Mat overlay = cv::imread("Traffic Objects/duck/duck.png", cv::IMREAD_COLOR);
    if (overlay.empty()) {
        std::cerr << "Error: Could not load overlay image." << std::endl;
        return -1;
    }

    // Loop to process video frames
    while (true) {
        // Capture a frame from the camera
        cv::Mat frame;
        cap >> frame;

        // Check if the frame is empty
        if (frame.empty()) {
            std::cerr << "Error: Failed to capture frame." << std::endl;
            break;
        }

        // Detect ArUco markers
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
        cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        cv::aruco::ArucoDetector detector(dictionary, detectorParams);
        detector.detectMarkers(frame, markerCorners, markerIds, rejectedCandidates);

        // Overlay the image on each detected ArUco marker
        for (size_t i = 0; i < markerIds.size(); ++i) {
            // Get the bounding box of the detected marker
            cv::Rect markerRect = cv::boundingRect(markerCorners[i]);

            // Resize the overlay image to match the size of the detected marker
            cv::Mat resizedOverlay;
            cv::resize(overlay, resizedOverlay, markerRect.size());

            // Overlay the resized image on the marker
            resizedOverlay.copyTo(frame(markerRect));
        }

        // Display the result
        cv::imshow("Result", frame);

        // Check for the 'q' key to exit
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the camera and close all windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
