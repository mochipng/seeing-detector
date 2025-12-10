#ifndef TRACKER_CGO_H
#define TRACKER_CGO_H

#ifdef __cplusplus
extern "C" {
#endif

int initializeTracker(int width, int height, const char* videoPath, const char* modelPath);
void processFrame(unsigned char* buffer);
int checkAudioPlay();
void shutdownTracker();

#ifdef __cplusplus
}
#endif

#endif