#ifndef TRACKER_H
#define TRACKER_H

extern int initializeTracker(int width, int height, const char* videoPath, const char* modelPath);
extern void processFrame(unsigned char* buffer);
extern void shutdownTracker();

#endif TRACKER_H