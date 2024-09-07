#ifndef CIRCLE_COORDINATES_H
#define CIRCLE_COORDINATES_H

// Define the number of coordinates (360 degrees)
#define NUM_COORDS 360
#define BINS 1000

#define GRP_DELAY 62.3

// Define the constants for the LFM signal
#define FS 100000.0
#define F_START 10000.0
#define F_STOP 30000.0
#define SWEEP_LENGTH 0.001
#define AMP 1.0

#define SPEED_OF_SOUND 344.351

// Pixel space attributes
const double x_min = -0.2;
const double x_max = 0.2;
const double y_min = -0.2;
const double y_max = 0.2;
const double z_value = 0.0;
const int num_points_per_side = 150;

// Define constants for transmitter and receiver coordinates
const float TX_INITIAL[3] = {0.0, 0.63, 0.23};
const float RX_INITIAL[3] = {0.0, 0.63, 0.22};

// Define the radii for the transmitter and receiver circles
const float RADIUS_TX = 0.63;
const float RADIUS_RX = 0.63;

const int NUM_PIXELS = 22500;

inline std::string background_dir = "/home/omkar/Desktop/Projetcs/Dome_experiments/2D_data/Test_run_background/2024_08_03-15_43_20";
inline std::string signal_dir = "/home/omkar/Desktop/Projetcs/Dome_experiments/2D_data/Test_run_360/2024_08_03-16_01_05";;
inline std::string image_file = "/home/omkar/Desktop/Projetcs/Dome_experiments/output_image.csv";

#endif // CIRCLE_COORDINATES_H