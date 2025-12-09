#ifndef TRACKER_H
#define TRACKER_H

#ifdef __cplusplus
extern "C" {
#endif

int initializeTracker(int width, int height, const char* videoPath, const char* modelPath);
void processFrame(unsigned char* buffer);
void shutdownTracker();

#ifdef __cplusplus
}
#endif

#endif