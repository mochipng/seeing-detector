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
bool audioPlay = false;

// cache last known face position for emoji overlay during video
dlib::full_object_detection lastShape;
std::vector<dlib::rectangle> lastFaces;
bool hasLastDetection = false;

// Freeze frame when user looks away
cv::Mat frozenFrame;
bool hasFrozenFrame = false;

// Delay counter for triggering video
int lookAwayCounter = 0;
const int LOOK_AWAY_THRESHOLD = 1;  // Trigger after 1 processed frame of looking away

// Delay audio trigger until video overlay is visible
int audioDelayCounter = 0;
const int AUDIO_DELAY = 5;  // Wait 5 frames after video starts before triggering audio

// frame skipping for performance
int frameSkipCounter = 0;
const int FRAME_SKIP = 2;  // Process every 2nd frame

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

// calculate eye aspect ratio
double getEyeAspectRatio(const dlib::full_object_detection& shape, int startIndex) {
    cv::Point2f p2(shape.part(startIndex + 1).x(), shape.part(startIndex + 1).y());
    cv::Point2f p3(shape.part(startIndex + 2).x(), shape.part(startIndex + 2).y());
    cv::Point2f p4(shape.part(startIndex + 4).x(), shape.part(startIndex + 4).y());
    cv::Point2f p5(shape.part(startIndex + 5).x(), shape.part(startIndex + 5).y());
    cv::Point2f p1(shape.part(startIndex).x(), shape.part(startIndex).y());
    cv::Point2f p6(shape.part(startIndex + 3).x(), shape.part(startIndex + 3).y());
    
    double verticalDist1 = cv::norm(p2 - p4);
    double verticalDist2 = cv::norm(p3 - p5);
    double horizontalDist = cv::norm(p1 - p6);
    
    if (horizontalDist == 0) return 0.3;
    
    return (verticalDist1 + verticalDist2) / (2.0 * horizontalDist);
}

// calculate gaze ratio (pupil position in eye)
double getGazeRatio(const dlib::full_object_detection& shape, int startIndex, const cv::Mat& frame) {
    cv::Point2f leftCorner(shape.part(startIndex).x(), shape.part(startIndex).y());
    cv::Point2f rightCorner(shape.part(startIndex + 3).x(), shape.part(startIndex + 3).y());
    
    std::vector<cv::Point> eyePoints;
    for (int i = 0; i < 6; i++) {
        eyePoints.push_back(cv::Point(shape.part(startIndex + i).x(), 
                                      shape.part(startIndex + i).y()));
    }
    cv::Rect eyeRect = cv::boundingRect(eyePoints);
    
    eyeRect.x = std::max(0, eyeRect.x - 5);
    eyeRect.y = std::max(0, eyeRect.y - 5);
    eyeRect.width = std::min(frame.cols - eyeRect.x, eyeRect.width + 10);
    eyeRect.height = std::min(frame.rows - eyeRect.y, eyeRect.height + 10);
    
    if (eyeRect.width <= 0 || eyeRect.height <= 0) return 0.5;
    
    cv::Mat eyeROI = frame(eyeRect);
    cv::Mat grayEye;
    cv::cvtColor(eyeROI, grayEye, cv::COLOR_BGR2GRAY);
    
    cv::GaussianBlur(grayEye, grayEye, cv::Size(5, 5), 0);
    cv::threshold(grayEye, grayEye, 30, 255, cv::THRESH_BINARY_INV);
    
    cv::Moments m = cv::moments(grayEye, true);
    if (m.m00 == 0) return 0.5;
    
    double pupilX = m.m10 / m.m00 + eyeRect.x;
    double eyeWidth = std::abs(rightCorner.x - leftCorner.x);
    if (eyeWidth < 1) return 0.5;
    
    double gazeRatio = (pupilX - leftCorner.x) / eyeWidth;
    
    return std::max(0.0, std::min(1.0, gazeRatio));
}

// functions to be called by Go

extern "C" int initializeTracker(int width, int height, const char* videoPath, const char* modelPath) {
    WIDTH = width;
    HEIGHT = height;

    detector = dlib::get_frontal_face_detector();

    try {
        dlib::deserialize(modelPath) >> predictor;
    } catch (dlib::serialization_error& e) {
        std::cerr << "Error loading shape predictor model: " << e.what() << std::endl;
        return 0;
    }

    if (!cap.open(0)) {
        std::cerr << "Error opening camera." << std::endl;
        return 0;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT);

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

    std::cout << "Tracker initialized successfully." << std::endl;
    return 1;
}

extern "C" void processFrame(unsigned char* buffer) {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) return;

    cv::resize(frame, frame, cv::Size(WIDTH, HEIGHT));

    const int MAX_VIDEO_FRAMES = 200;
    
    // Frame skipping for performance
    frameSkipCounter++;
    bool shouldProcess = (frameSkipCounter % FRAME_SKIP == 0);
    
    // Only do face detection and gaze tracking when video is NOT playing
    if (videoFrameCount == 0 && shouldProcess) {
        dlib::cv_image<dlib::bgr_pixel> img(frame);
        std::vector<dlib::rectangle> faces = detector(img);
        
        if (!faces.empty()) {
            dlib::full_object_detection shape = predictor(img, faces[0]);
            
            // Cache for emoji positioning if video starts
            lastShape = shape;
            lastFaces = faces;
            hasLastDetection = true;
            
            // Gaze tracking - REMOVED eye closure check
            const double GAZE_CENTER_MIN = 0.30;
            const double GAZE_CENTER_MAX = 0.70;

            double leftGaze = getGazeRatio(shape, 36, frame);
            double rightGaze = getGazeRatio(shape, 42, frame);
            double avgGaze = (leftGaze + rightGaze) / 2.0;

            bool lookingAway = (avgGaze < GAZE_CENTER_MIN) || (avgGaze > GAZE_CENTER_MAX);
            
            userLooking = !lookingAway;

            static int frameDebugCounter = 0;
            if (++frameDebugCounter % 30 == 0) {
                std::cout << "L:" << leftGaze << " R:" << rightGaze 
                << " Looking:" << userLooking << std::endl;
            }
        } else {
            userLooking = false;
        }
        
        // Delay logic: increment counter when looking away, reset when looking
        if (!userLooking) {
            lookAwayCounter++;
        } else {
            lookAwayCounter = 0;
        }
        
        // Start video only after threshold is reached
        if (lookAwayCounter >= LOOK_AWAY_THRESHOLD) {
            std::cout << "TRIGGERING VIDEO!" << std::endl;
            videoFrameCount = 1;
            audioDelayCounter = 0;  // Start audio delay counter
            lookAwayCounter = 0;
            
            // Freeze the current frame
            frozenFrame = frame.clone();
            hasFrozenFrame = true;
        }
    } else if (videoFrameCount > 0 && shouldProcess) {
        // Video is playing - check if user looked back to stop it
        dlib::cv_image<dlib::bgr_pixel> img(frame);
        std::vector<dlib::rectangle> faces = detector(img);
        
        if (!faces.empty()) {
            dlib::full_object_detection shape = predictor(img, faces[0]);
            
            const double GAZE_CENTER_MIN = 0.30;
            const double GAZE_CENTER_MAX = 0.70;

            double leftGaze = getGazeRatio(shape, 36, frame);
            double rightGaze = getGazeRatio(shape, 42, frame);
            double avgGaze = (leftGaze + rightGaze) / 2.0;

            bool lookingAway = (avgGaze < GAZE_CENTER_MIN) || (avgGaze > GAZE_CENTER_MAX);
            
            userLooking = !lookingAway;
            
            if (userLooking) {
                std::cout << "VIDEO STOPPED - User looking back" << std::endl;
                videoFrameCount = 0;
                videoOverlay.set(cv::CAP_PROP_POS_FRAMES, 0);
                hasFrozenFrame = false;
            }
        }
    }
    
    // If video is playing, use the frozen frame instead of live camera
    if (videoFrameCount > 0 && hasFrozenFrame) {
        frame = frozenFrame.clone();
    }

    // Display emoji if video is playing and we have cached face position
    if (videoFrameCount > 0 && hasLastDetection) {
        cv::Point leftEyeCenter(
            (lastShape.part(36).x() + lastShape.part(39).x()) / 2,
            (lastShape.part(36).y() + lastShape.part(39).y()) / 2
        );
        cv::Point rightEyeCenter(
            (lastShape.part(42).x() + lastShape.part(45).x()) / 2,
            (lastShape.part(42).y() + lastShape.part(45).y()) / 2
        );
        cv::Point eyesMidpoint(
            (leftEyeCenter.x + rightEyeCenter.x) / 2,
            (leftEyeCenter.y + rightEyeCenter.y) / 2
        );

        int emojiWidth = static_cast<int>(lastFaces[0].width() * 1.0);
        int emojiHeight = static_cast<int>(lastFaces[0].width() * 0.7);
        cv::Size emojiSize(emojiWidth, emojiHeight);

        overlayImage(frame, emojiImage, eyesMidpoint, emojiSize);
    }

    // Video overlay
    if (videoFrameCount > 0) {
        videoFrameCount++;
        
        // Trigger audio after delay
        if (videoFrameCount == AUDIO_DELAY) {
            audioPlay = true;
            std::cout << "AUDIO TRIGGERED" << std::endl;
        }

        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

        cv::Mat videoFrame;
        videoOverlay >> videoFrame;

        if (videoFrame.empty() || videoFrameCount >= MAX_VIDEO_FRAMES) {
            std::cout << "VIDEO ENDED" << std::endl;
            videoFrameCount = 0;
            videoOverlay.set(cv::CAP_PROP_POS_FRAMES, 0);
        } else {
            cv::resize(videoFrame, videoFrame, frame.size());
            
            cv::Mat mask;
            cv::inRange(videoFrame, cv::Scalar(0, 100, 0), cv::Scalar(100, 255, 100), mask);
            
            cv::Mat inverseMask;
            cv::bitwise_not(mask, inverseMask);
            
            videoFrame.copyTo(frame, inverseMask);
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
    audioPlay = false;
    return status ? 1 : 0;
}