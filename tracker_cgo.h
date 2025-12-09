#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif

int initializeTracker(int width, int height, const char* videoPath, const char* modelPath);
void processFrame(unsigned char* buffer);
void overlayImage(cv::Mat& background, const cv::Mat& foreground, cv::Point location, cv::Size targetSize);
void shutdownTracker();

#ifdef __cplusplus
}
#endif

#endif