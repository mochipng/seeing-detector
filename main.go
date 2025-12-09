package main

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: ./libtracker.so
#include "tracker_cgo.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"os"
	"unsafe"
	
	"github.com/veandco/go-sdl2/sdl"
)

// define constant variables
const (
	width  = 640
	height = 480
	buffer = width * height * 3
)

func main() {
	// file names
	video := "distraction_video.mp4"
	emoji := "emoji.png"
	model := "shape_predictor_68_face_landmarks.dat"

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
		texture.Update(nil, unsafe.Pointer(&frameBuffer[0]), width*3)

		renderer.Clear()
		renderer.Copy(texture, nil, nil)
		renderer.Present()

		sdl.Delay(1)
	}

}
