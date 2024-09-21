#ifndef BEAMFORMER_CUH
#define BEAMFORMER_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <thrust/complex.h>
#include "auxilary.h"
#include "_readsignal.h"
#include "_signalprocessing.h"

std::vector<std::vector<double>> createPixelSpace(double x_min, double x_max, double y_min, double y_max,
                                                     int num_points_per_side, double z_value);

std::vector<std::vector<double>> GenerateSensorCoordinates(double radius, double z);

std::vector<std::complex<double>> _beamform_extern(std::vector<double> h_pixels_flat, std::vector<double> h_tx_coords, std::vector<double> h_rx_coords, 
                                            std::vector<std::complex<double>> h_x_analytic_flat, int num_pixels,
                                            int num_coords, double c, double Fs, int signal_length);

__global__ void beamforming_kernel(double *pixels, double *tx_coords, double *rx_coords, 
                                   thrust::complex<double> *x_analytic, thrust::complex<double> *output, 
                                   int num_coords, double c, double Fs, int signal_length);

__global__ void generate_circle_coordinates_kernel(double *tx_coords, double *rx_coords, 
                                                   double radius_tx, double radius_rx, 
                                                   double z_tx, double z_rx);

#endif // BEAMFORMER_CUH