package main

/*
#cgo CXXFLAGS: -std=c++17
#cgo CXXFLAGS: -I/opt/homebrew/opt/opencv/include/opencv4
#cgo LDFLAGS: -L/opt/homebrew/opt/opencv/lib
#cgo LDFLAGS: -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs
#cgo LDFLAGS: -ldlib
#cgo LDFLAGS: -framework Accelerate
#cgo LDFLAGS: -L${SRCDIR} -ltracker

#include "tracker_cgo.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"io"
	"log"
	"os"
	"unsafe"

	"github.com/hajimehoshi/go-mp3"
	"github.com/hajimehoshi/oto"
	"github.com/veandco/go-sdl2/sdl"
)

// define constant variables
const (
	width  = 640
	height = 480
	buffer = width * height * 3
)

var audioPlayer *oto.Player
var audioContext *oto.Context

var audioData []byte
var audioSampleRate int

func loadAudio(filePath string) {
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("Failed to open audio file: %v", err)
	}
	defer file.Close()

	decoder, err := mp3.NewDecoder(file)
	if err != nil {
		log.Fatalf("Failed to create MP3 decoder: %v", err)
	}

	audioSampleRate = decoder.SampleRate()
	audioData, err = io.ReadAll(decoder)
	if err != nil {
		log.Fatalf("Failed to read audio data: %v", err)
	}

	audioContext, err = oto.NewContext(audioSampleRate, 2, 2, 8192)
	if err != nil {
		log.Fatalf("Failed to create Oto context: %v", err)
	}

	audioPlayer = audioContext.NewPlayer()
}

func playAudio() {
	if audioPlayer != nil {
		audioPlayer.Close()
	}

	// Get the context that was already created in loadAudio
	audioPlayer = audioContext.NewPlayer()

	_, err := audioPlayer.Write(audioData)
	if err != nil {
		log.Printf("Failed to play audio: %v", err)
	}
}

func main() {
	// file names
	video := "distraction_video.mp4"
	emoji := "emoji.png"
	model := "shape_predictor_68_face_landmarks.dat"

	loadAudio("carl.mp3")

	cVideo := C.CString(video)
	defer C.free(unsafe.Pointer(cVideo))

	cEmoji := C.CString(emoji)
	defer C.free(unsafe.Pointer(cEmoji))

	cModel := C.CString(model)
	defer C.free(unsafe.Pointer(cModel))

	success := C.initializeTracker(
		C.int(width),
		C.int(height),
		cVideo,
		cModel,
	)

	if success == 0 {
		fmt.Println("Failed to initialize tracker")
		os.Exit(1)
	}

	defer C.shutdownTracker()

	if err := sdl.Init(sdl.INIT_EVERYTHING); err != nil {
		panic(fmt.Sprintf("Failed to initialize SDL: %v", err))
	}

	defer sdl.Quit()

	window, err := sdl.CreateWindow("Seeing Detector", sdl.WINDOWPOS_UNDEFINED, sdl.WINDOWPOS_UNDEFINED, int32(width), int32(height), sdl.WINDOW_SHOWN)
	if err != nil {
		panic(fmt.Sprintf("Failed to create window: %v", err))
	}

	defer window.Destroy()

	renderer, err := sdl.CreateRenderer(window, -1, sdl.RENDERER_ACCELERATED)
	if err != nil {
		panic(fmt.Sprintf("Failed to create renderer: %v", err))
	}

	defer renderer.Destroy()

	texture, err := renderer.CreateTexture(sdl.PIXELFORMAT_BGR24, sdl.TEXTUREACCESS_STREAMING, int32(width), int32(height))
	if err != nil {
		panic(fmt.Sprintf("Failed to create texture: %v", err))
	}

	defer texture.Destroy()

	running := true
	frameBuffer := make([]byte, buffer)

	// main loop
	for running {
		for event := sdl.PollEvent(); event != nil; event = sdl.PollEvent() {
			if _, ok := event.(*sdl.QuitEvent); ok {
				running = false
			}
		}

		C.processFrame((*C.uchar)(unsafe.Pointer(&frameBuffer[0])))

		if C.checkAudioPlay() != 0 {
			playAudio()
		}

		texture.Update(nil, unsafe.Pointer(&frameBuffer[0]), width*3)

		renderer.Clear()
		renderer.Copy(texture, nil, nil)
		renderer.Present()

		sdl.Delay(1)
	}

}
