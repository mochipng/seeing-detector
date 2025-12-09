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
dlib::frontal_face_detector detector;
dlib::shape_predictor predictor;
cv::VideoCapture cap;
cv::VideoCapture videoOverlay;
cv::Mat emojiImage;

// camera dimensions, user state
int WIDTH = 640;
int HEIGHT = 480;
bool userLooking = true;

// convert OpenCV to dlib image format
dlib::cv_image<dlib::bgr_pixel> dlib_img(cv::Mat& mat) {
    return dlib::cv_image<dlib::bgr_pixel>(mat);
}

// overlay emoji on frame
void overlayImage(cv::Mat& background, const cv::Mat& foreground, cv::Point location) {
    if (foreground.channels() != 4 || background.channels() != 3) {
        std::cerr << "Foreground image does not have an alpha channel." << std::endl;
        return;
    }

    cv::Size targetSize(60, 60);
    cv::Mat resizedEmoji;
    cv::resize(foreground, resizedEmoji, targetSize, 0, 0, cv::INTER_AREA);

    int fx = location.x - targetSize.width / 2;
    int fy = location.y - targetSize.height - 10;

    int xStart = std::max(fx, 0);
    int yStart = std::max(fy, 0);
    int xEnd = std::min(background.cols, fx + resizedEmoji.cols);
    int yEnd = std::min(background.rows, fy + resizedEmoji.rows);

    for (int y = yStart; y < yEnd; ++y) {
        for (int x = xStart; x < xEnd; ++x) {
            int ejX = x - fx;
            int ejY = y - fy;

            cv::Vec4b ejPixel = resizedEmoji.at<cv::Vec4b>(ejY, ejX);
            cv::Vec3b& bgPixel = background.at<cv::Vec3b>(y, x);

            double alpha = ejPixel[3] / 255.0f;

            for (int c = 0; c < 3; ++c) {
                bgPixel[c] = static_cast<uchar>(bgPixel[c] * (1.0f - alpha) + ejPixel[c] * alpha);
            }
        }

    }
}

// calculate horizontal eye poisition relative to width

double getGazeRatio(const dlib::full_object_detection& shape, int startIndex) {
    cv::Point p1(shape.part(startIndex).x(), shape.part(startIndex).y());
    cv::Point p4(shape.part(startIndex + 3).x(), shape.part(startIndex + 3).y());

    double gazePointX = (shape.part(startIndex + 1).x() + shape.part(startIndex + 5).x() ) / 2.0;
    
    double eyeWidth = std::abs(p4.x - p1.x);
    if (eyeWidth == 0) return 0.5;

    return (gazePointX - p1.x) / eyeWidth;
}

// functions to be called by Go

extern "C" int initializeTracker(int width, int height, const char* videoPath, const char* modelPath) {
    WIDTH = width;
    HEIGHT = height;

    // load shape predictor from dlib
    detector = dlib::get_frontal_face_detector();

    try {
        dlib::deserialize(modelPath) >> predictor;
    } catch (dlib::serialization_error& e) {
        std::cerr << "Error loading shape predictor model." << e.what() << std::endl;
        return 0;
    }

    // initialize camera
    if (!cap.open(0)) {
        std::cerr << "Error when trying to open camera." << std::endl;
        return 0;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT);

    // load video overlays
    videoOverlay.open(videoPath);
    if (!videoOverlay.isOpened())
        std::cerr << "Error loading overlay video." << std::endl;

    emojiImage = cv::imread("emoji.png", cv::IMREAD_UNCHANGED);
    if (emojiImage.empty())
        std::cerr << "Error loading emoji image." << std::endl;

    // function successful
    std::cout << "Tracker initialized successfully." << std::endl;
    return 1;
}

extern "C" void processFrame(unsigned char* buffer) {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) return;

    cv::resize(frame, frame, cv::Size(WIDTH, HEIGHT));
    dlib::cv_image<dlib::bgr_pixel> img(frame);

    std::vector<dlib::rectangle> faces = detector(img);
    dlib::full_object_detection shape;

    if (!faces.empty()) {
        shape = predictor(img, faces[0]);

        double leftGaze = getGazeRatio(shape, 42);
        double rightGaze = getGazeRatio(shape, 36);

        const double THRESHOLD_OUTER = 0.35;
        const double THRESHOLD_INNER = 0.65;

        bool rightGazing = rightGaze < THRESHOLD_OUTER || rightGaze > THRESHOLD_INNER;
        bool leftGazing = leftGaze < THRESHOLD_OUTER || leftGaze > THRESHOLD_INNER;
        
        userLooking = !(rightGazing || leftGazing);

    } else {
        userLooking = false;
    }

    if (!userLooking) {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

        cv::Mat videoFrame;
        videoOverlay >> videoFrame;

        if (videoFrame.empty()) {
            videoOverlay.set(cv::CAP_PROP_POS_FRAMES, 0);
            videoOverlay >> videoFrame;
        }

        cv::resize(videoFrame, videoFrame, frame.size());

        cv::addWeighted(frame, 0.7, videoFrame, 0.3, 0.0, frame);

        if (!faces.empty()) {
            cv::Point headPosition(shape.part(30).x(), shape.part(30).y());
            overlayImage(frame, emojiImage, headPosition);
        }
    }

    if (frame.total() * frame.elemSize() <= WIDTH * HEIGHT * 3) {
            std::memcpy(buffer, frame.data, frame.total() * frame.elemSize());
        }
}

extern "C" void shutdownTracker() {
    if (cap.isOpened()) {
        cap.release();
    }

    if (videoOverlay.isOpened()) {
        videoOverlay.release();
    }
}