// dependencies
#include "tracker.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

// global variables
dlib::get_frontal_face_detector detector;
dlib::shape_predictor predictor;
cv::VideoCapture cap;
cv::VideoCapture videoOverlay;
cv::Mat emojiImage;

// camera dimensions, user state
int WIDTH = 640;
int HEIGHT = 480;
bool userLooking = true;

extern "C" int initializeTracker(int width, int height, const char* videoPath, const char* modelPath) {
    WIDTH = width;
    HEIGHT = height;

    // load shape predictor from dlib
    dlib = dlib::shape_predictor();
    try {
        dlib::desteralize(modelPath) >> predictor;
    } catch {
        std::cerr << "Error loading shape predictor model." << std::endl;
        return 0;
    }

    // initialize camera
    if (!cap.open(0)) {
        std::cer << "Error when trying to open camera." << std::endl;
        return 0;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT);

    // load video overlays
    overlayVideo.open(videoPath);
    if (!overlayVideo.isOpened())
        std::cerr << "Error loading overlay video." << std::endl;

    emojiImage = cv::imread("emoji.png", cv::IMREAD_UNCHANGED);
    if (emojiImage.empty())
        std::cerr << "Error loading emoji image." << std::endl;

    // function successful
    std::cout << "Tracker initialized successfully." << std::endl;
    return 1;

}