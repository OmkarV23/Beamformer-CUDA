#ifndef CIRCLE_COORDINATES_H
#define CIRCLE_COORDINATES_H

// Define the number of coordinates (360 degrees)
#define NUM_COORDS 360
#define BINS 1000

#define GRP_DELAY 62.3

#define FS 100000.0
#define F_START 10000.0
#define F_STOP 30000.0
#define SWEEP_LENGTH 0.001
#define AMP 1.0

// Define constants for transmitter and receiver coordinates
const float TX_INITIAL[3] = {0.0f, 0.63f, 0.23f};
const float RX_INITIAL[3] = {0.0f, 0.63f, 0.22f};

// Define the radii for the transmitter and receiver circles
const float RADIUS_TX = 0.63f;
const float RADIUS_RX = 0.63f;

const int NUM_PIXELS = 22500;

#endif // CIRCLE_COORDINATES_H