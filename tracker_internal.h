#ifndef TRACKER_INTERNAL_H
#define TRACKER_INTERNAL_H

#include <opencv2/opencv.hpp>

// Internal C++ functions (not exposed to Go)
void overlayImage(cv::Mat& background, const cv::Mat& foreground, cv::Point location, cv::Size targetSize);

#endif