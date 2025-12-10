// dependencies
#include "tracker_cgo.h"
#include "tracker_internal.h"
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

// camera dimensions, user state, counters for video overlay frames and frames user is looking away
int WIDTH = 640;
int HEIGHT = 480;
bool userLooking = false;
int videoFrameCount = 0;
int distractionCounter = 0;
bool audioPlay = false;
bool videoHasPlayed = false;

// convert OpenCV to dlib image format
dlib::cv_image<dlib::bgr_pixel> dlib_img(cv::Mat& mat) {
    return dlib::cv_image<dlib::bgr_pixel>(mat);
}

// overlay emoji on frame
void overlayImage(cv::Mat& background, const cv::Mat& foreground, cv::Point location, cv::Size targetSize) {
    if (foreground.channels() != 4 || background.channels() != 3) {
        std::cerr << "Foreground image does not have an alpha channel." << std::endl;
        return;
    }

    cv::Mat resizedEmoji;
    cv::resize(foreground, resizedEmoji, targetSize, 0, 0, cv::INTER_AREA);

    // Adjusted vertical offset (+50) to center the emoji better on the face
    int fx = location.x - targetSize.width / 2;
    int fy = location.y - targetSize.height / 2 - 20;

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
double getEyeAspectRatio(const dlib::full_object_detection& shape, int startIndex) {
    // Vertical eye landmarks
    cv::Point2f p2(shape.part(startIndex + 1).x(), shape.part(startIndex + 1).y());
    cv::Point2f p3(shape.part(startIndex + 2).x(), shape.part(startIndex + 2).y());
    cv::Point2f p4(shape.part(startIndex + 4).x(), shape.part(startIndex + 4).y());
    cv::Point2f p5(shape.part(startIndex + 5).x(), shape.part(startIndex + 5).y());
    
    // horizontal eye landmarks
    cv::Point2f p1(shape.part(startIndex).x(), shape.part(startIndex).y());
    cv::Point2f p6(shape.part(startIndex + 3).x(), shape.part(startIndex + 3).y());
    
    double verticalDist1 = cv::norm(p2 - p4);
    double verticalDist2 = cv::norm(p3 - p5);
    double horizontalDist = cv::norm(p1 - p6);
    
    if (horizontalDist == 0) return 0.3;
    
    return (verticalDist1 + verticalDist2) / (2.0 * horizontalDist);
}

double getGazeRatio(const dlib::full_object_detection& shape, int startIndex) {
    // eye corner points
    cv::Point2f leftCorner(shape.part(startIndex).x(), shape.part(startIndex).y());
    cv::Point2f rightCorner(shape.part(startIndex + 3).x(), shape.part(startIndex + 3).y());
    
    // iris/pupil approximation (center of eye)
    double pupilX = (shape.part(startIndex + 1).x() + shape.part(startIndex + 2).x() + 
                     shape.part(startIndex + 4).x() + shape.part(startIndex + 5).x()) / 4.0;
    
    double eyeWidth = std::abs(rightCorner.x - leftCorner.x);
    if (eyeWidth < 1) return 0.5;

    // normalize pupil position relative to eye width
    double gazeRatio = (pupilX - leftCorner.x) / eyeWidth;
    
    return gazeRatio;
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
    if (!videoOverlay.isOpened()) {
        std::cerr << "Error loading overlay video." << std::endl;
        return 0;
    }

    emojiImage = cv::imread("emoji.png", cv::IMREAD_UNCHANGED);
    if (emojiImage.empty()) {
        std::cerr << "Error loading emoji image." << std::endl;
        return 0;
    }

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

        double leftGaze = getGazeRatio(shape, 36);
        double rightGaze = getGazeRatio(shape, 42); 
        
        // calculate eye aspect ratio to detect closed eyes
        double leftEAR = getEyeAspectRatio(shape, 36);
        double rightEAR = getEyeAspectRatio(shape, 42);
        double avgEAR = (leftEAR + rightEAR) / 2.0;
        
        const double GAZE_CENTER_MIN = 0.43;
        const double GAZE_CENTER_MAX = 0.57;
        const double EAR_THRESHOLD = 0.23;
        
        // check if eyes are closed
        bool eyesClosed = avgEAR < EAR_THRESHOLD;
        
        // check if looking away horizontally
       // Check EITHER eye looking away (OR not AND)
        bool leftLookingAway = (leftGaze < GAZE_CENTER_MIN) || (leftGaze > GAZE_CENTER_MAX);
        bool rightLookingAway = (rightGaze < GAZE_CENTER_MIN) || (rightGaze > GAZE_CENTER_MAX);
        bool lookingAway = leftLookingAway || rightLookingAway;

        userLooking = !eyesClosed && !lookingAway;

        // Debug output every 30 frames
        static int frameDebugCounter = 0;
        if (++frameDebugCounter % 30 == 0) {
            std::cout << "L:" << leftGaze << " R:" << rightGaze << " EAR:" << avgEAR 
              << " Looking:" << userLooking << " Counter:" << distractionCounter << std::endl;
        }

    } else {
        // If face is lost (e.g., severe head turn), trigger the effect
        userLooking = false;
    }

    const int MAX_VIDEO_FRAMES = 400;
    const int DISTRACTION_THRESHOLD = 2; 
    
    // Debounce Logic: Count up/down the distraction state
    if (!userLooking) {
        distractionCounter = std::min(distractionCounter + 1, DISTRACTION_THRESHOLD);
    } else {
        distractionCounter = std::max(distractionCounter - 1, 0); 

        if (videoFrameCount > 0) {
            videoFrameCount = 0;
            videoOverlay.set(cv::CAP_PROP_POS_FRAMES, 0);
            std::cout << "VIDEO STOPPED - User looking back" << std::endl;
        }
    }

    // only trigger if video hasn't played yet
    bool shouldStartVideo = (distractionCounter >= DISTRACTION_THRESHOLD) && 
                           (videoFrameCount == 0) && 
                           !videoHasPlayed;
    
    if (shouldStartVideo) {
        std::cout << "TRIGGERING VIDEO!" << std::endl;
        audioPlay = true;
        videoHasPlayed = true;
        videoFrameCount = 1;
    }

        if (!faces.empty() && distractionCounter > 0) {
        cv::Point leftEyeCenter(
        (shape.part(36).x() + shape.part(39).x()) / 2,
        (shape.part(36).y() + shape.part(39).y()) / 2
        );
        cv::Point rightEyeCenter(
        (shape.part(42).x() + shape.part(45).x()) / 2,
        (shape.part(42).y() + shape.part(45).y()) / 2
        );
        cv::Point eyesMidpoint(
        (leftEyeCenter.x + rightEyeCenter.x) / 2,
        (leftEyeCenter.y + rightEyeCenter.y) / 2
        );
        
        // dynamic scaling based on face width
        int faceWidth = faces[0].width();
        int emojiWidth = static_cast<int>(faceWidth * 1.0);
        int emojiHeight = static_cast<int>(faceWidth * 0.7);
        cv::Size emojiSize(emojiWidth, emojiHeight); 

        overlayImage(frame, emojiImage, eyesMidpoint, emojiSize);
    }
    
    bool videoIsRunning = videoFrameCount > 0;
    bool showOverlay = videoIsRunning;

    if (showOverlay) {
        
        videoFrameCount = std::min(videoFrameCount + 1, MAX_VIDEO_FRAMES); 

        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

        cv::Mat videoFrame;
        videoOverlay >> videoFrame;

        if (videoFrame.empty() || videoFrameCount >= MAX_VIDEO_FRAMES) {
            if (videoFrameCount >= MAX_VIDEO_FRAMES) {
                std::cout << "VIDEO ENDED" << std::endl;
                videoFrameCount = 0; 
                distractionCounter = 0;
                videoHasPlayed = false;
                videoOverlay.set(cv::CAP_PROP_POS_FRAMES, 0); 
            }
        }

        if (!videoFrame.empty()) {
            cv::resize(videoFrame, videoFrame, frame.size()); 

            cv::Scalar lowerGreen = cv::Scalar(0, 100, 0);
            cv::Scalar upperGreen = cv::Scalar(100, 255, 100);

            cv::Mat mask;
            cv::inRange(videoFrame, lowerGreen, upperGreen, mask);
            
            cv::Mat inverseMask;
            cv::bitwise_not(mask, inverseMask);

            cv::Mat foregroundContent;
            videoFrame.copyTo(foregroundContent, inverseMask);

            cv::Mat backgroundContent;
            frame.copyTo(backgroundContent, mask);

            cv::add(foregroundContent, backgroundContent, frame);
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

extern "C" int checkAudioPlay() {
    bool status = audioPlay;
    audioPlay = false; // Reset after checking
    return status ? 1 : 0;
}