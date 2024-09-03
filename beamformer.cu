#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <thrust/complex.h>
// #include <opencv2/opencv.hpp>
#include "auxilary.h"
#include "_readsignal.h"
#include "_signalprocessing.h"

__global__ void beamforming_kernel(float *pixels, float *tx_coords, float *rx_coords, 
                                   thrust::complex<float> *x_analytic, float *output, 
                                   int num_coords, float c, float Fs, int signal_length) {
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    //print pixels
    // printf("Pixel: %f, %f, %f\n", pixels[0], pixels[1], pixels[2]);

    if (pixel_idx < 22500) {  // 150x150 = 22500 pixels
        float d1, d2, tof;
        thrust::complex<float> sum = thrust::complex<float>(0.0f, 0.0f);
        
        for (int i = 0; i < num_coords; i++) {
            d1 = sqrtf(powf(pixels[3 * pixel_idx] - tx_coords[3 * i], 2) + 
                       powf(pixels[3 * pixel_idx + 1] - tx_coords[3 * i + 1], 2) + 
                       powf(pixels[3 * pixel_idx + 2] - tx_coords[3 * i + 2], 2));
            
            d2 = sqrtf(powf(pixels[3 * pixel_idx] - rx_coords[3 * i], 2) + 
                       powf(pixels[3 * pixel_idx + 1] - rx_coords[3 * i + 1], 2) + 
                       powf(pixels[3 * pixel_idx + 2] - rx_coords[3 * i + 2], 2));
            
            // printf("d1: %f, d2: %f\n", d1, d2);

            tof = ((d1 + d2) / c) * Fs;
            
            // Sinc interpolation and summing values at the pixels
            for (int j = 0; j < signal_length; j++) {
                float sinc_value = sinf(M_PI * (j - tof)) / (M_PI * (j - tof));
                sum += x_analytic[i * signal_length + j] * sinc_value;
                // printf("Sum: %f, Pixel_idx: %d\n", sum, pixel_idx);
            }
            output[pixel_idx] += thrust::abs(sum);
        }
        // Store only the magnitude of the complex sum
        //print sum and pixel_idx
        // output[pixel_idx] = thrust::abs(sum);
    }
}


__global__ void generate_circle_coordinates_kernel(float *tx_coords, float *rx_coords, 
                                                   float radius_tx, float radius_rx, 
                                                   float z_tx, float z_rx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < 360) {  // Loop over all angles (degrees) from 1 to 360
        float theta = (i + 1) * (M_PI / 180.0f);  // Convert angle to radians

        // Coordinates for the circle in XY plane (X, Y)
        float x_tx = radius_tx * cosf(theta);
        float y_tx = radius_tx * sinf(theta);

        float x_rx = radius_rx * cosf(theta);
        float y_rx = radius_rx * sinf(theta);

        // Set the coordinates in the tx_coords and rx_coords arrays
        tx_coords[i * 3] = x_tx;
        tx_coords[i * 3 + 1] = y_tx;
        tx_coords[i * 3 + 2] = z_tx;

        rx_coords[i * 3] = x_rx;
        rx_coords[i * 3 + 1] = y_rx;
        rx_coords[i * 3 + 2] = z_rx;
    }
}

std::vector<std::vector<float>> createPixelSpace(float x_min, float x_max, float y_min, float y_max, int num_points_per_side, float z_value) {
    // Calculate the step size for x and y directions
    float x_step = (x_max - x_min) / (num_points_per_side - 1);
    float y_step = (y_max - y_min) / (num_points_per_side - 1);

    // Initialize the pixel space with size (22500, 3)
    std::vector<std::vector<float>> pixel_space(num_points_per_side * num_points_per_side, std::vector<float>(3, 0.0f));

    // Fill the pixel space
    int index = 0;
    for (int i = 0; i < num_points_per_side; ++i) {
        for (int j = 0; j < num_points_per_side; ++j) {
            float x = x_min + i * x_step;
            float y = y_min + j * y_step;
            pixel_space[index][0] = x;
            pixel_space[index][1] = y;
            pixel_space[index][2] = z_value;
            ++index;
        }
    }

    return pixel_space;
}

// void saveOutputImageToPNG(const std::string& filename, float output_image[150][150]) {
//     cv::Mat img(150, 150, CV_32F, output_image);  // Convert array to cv::Mat

//     // Normalize image to [0, 255] if needed
//     cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
//     img.convertTo(img, CV_8U); // Convert to 8-bit for saving as PNG

//     // Save image
//     cv::imwrite(filename, img);
//     std::cout << "Output image saved to " << filename << std::endl;
// }

int main() {

    std::vector<double> LFM = generate_lfm_signal(F_START, F_STOP, FS, SWEEP_LENGTH, AMP);

    // std::cout << "LFM signal generated successfully!" << std::endl;

    std::string background_dir = "/home/omkar/Desktop/Projetcs/Dome_experiments/2D_data/Test_run_background/2024_08_03-15_43_20";
    std::vector<std::vector<double>> background = readAllCSVFiles(background_dir);

    std::vector<double> mean_background = computeMeanSignal(background);

    std::string signal_dir = "/home/omkar/Desktop/Projetcs/Dome_experiments/2D_data/Test_run_360/2024_08_03-16_01_05";
    std::vector<std::vector<double>> signal = readAllCSVFiles(signal_dir);

    // subtract the mean background from the signal
    std::vector<std::vector<double>> signal_minus_background = subtractMeanBackground(signal, mean_background);

    // std::cout << "Signal processing done successfully!" << std::endl;

    // std::string filename1 = "/home/omkar/Desktop/Projetcs/Dome_experiments/signal.csv";
    // saveToCSV(signal_minus_background[0], filename1);

    // account for group delay and do matched filtering
    std::vector<std::vector<std::complex<double>>> x_analytic;

    // Flatten x_analytic into a 1D array of thrust::complex<float>
    std::vector<thrust::complex<float>> h_x_analytic_flat;

    for (int i = 0; i < signal_minus_background.size(); i++) {
        std::vector<std::complex<double>> x_ana = match_filter(hilbert_transform(correctGroupDelay(signal_minus_background[i], GRP_DELAY, FS)), LFM);
        for (int j = 0; j < x_ana.size(); j++) {
            h_x_analytic_flat.emplace_back(static_cast<float>(x_ana[j].real()), static_cast<float>(x_ana[j].imag()));
        }
    }
    
    // The total size would be num_coords * bins (flattened size)
    size_t total_size = h_x_analytic_flat.size();

    float x_min = -0.2f;
    float x_max = 0.2f;
    float y_min = -0.2f;
    float y_max = 0.2f;
    int num_points_per_side = 150;  // 150x150 grid
    float z_value = 0.0f;

    // Create the pixel space
    std::vector<std::vector<float>> h_pixels = createPixelSpace(x_min, x_max, y_min, y_max, num_points_per_side, z_value);

    std::vector<float> h_pixels_flat;
    for (int i = 0; i < h_pixels.size(); ++i) {
        h_pixels_flat.push_back(h_pixels[i][0]);
        h_pixels_flat.push_back(h_pixels[i][1]);
        h_pixels_flat.push_back(h_pixels[i][2]);
    }

    
    // Allocate memory on the GPU (Device)
    float *d_pixels, *d_output;
    thrust::complex<float>* d_x_analytic;

    float *d_tx_coords, *d_rx_coords;
    cudaMalloc((void**)&d_tx_coords, NUM_COORDS * sizeof(TX_INITIAL) * sizeof(float));
    cudaMalloc((void**)&d_rx_coords, NUM_COORDS * sizeof(RX_INITIAL) * sizeof(float));


    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocks1 = (360 + threadsPerBlock - 1) / threadsPerBlock;
    generate_circle_coordinates_kernel<<<blocks1, threadsPerBlock>>>(d_tx_coords, d_rx_coords, 
                                                                            RADIUS_TX, RADIUS_RX, TX_INITIAL[2], RX_INITIAL[2]);

    ///////////////////////////////////////////////////////////////////////////////

    // // print tx_coords and rx_coords
    // float *h_tx_coords, *h_rx_coords;
    // h_tx_coords = (float*)malloc(NUM_COORDS * sizeof(TX_INITIAL) * sizeof(float));
    // h_rx_coords = (float*)malloc(NUM_COORDS * sizeof(RX_INITIAL) * sizeof(float));
    // if (h_tx_coords == NULL || h_rx_coords == NULL) {
    //     std::cerr << "Failed to allocate memory for h_tx_coords or h_rx_coords on the host." << std::endl;
    //     return -1;
    // }

    // // Copy tx_coords and rx_coords from device to host
    // cudaMemcpy(h_tx_coords, d_tx_coords, NUM_COORDS * sizeof(TX_INITIAL) * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_rx_coords, d_rx_coords, NUM_COORDS * sizeof(RX_INITIAL) * sizeof(float), cudaMemcpyDeviceToHost);

    // // print tx_coords and rx_coords
    // for (int i = 0; i < NUM_COORDS; i++) {
    //     printf("TX: %f, %f, %f\n", h_tx_coords[3 * i], h_tx_coords[3 * i + 1], h_tx_coords[3 * i + 2]);
    //     printf("RX: %f, %f, %f\n", h_rx_coords[3 * i], h_rx_coords[3 * i + 1], h_rx_coords[3 * i + 2]);
    // }

    ///////////////////////////////////////////////////////////////////////////////

    cudaMalloc((void**)&d_pixels, h_pixels_flat.size() * sizeof(float));
    cudaMalloc((void**)&d_x_analytic, total_size * sizeof(thrust::complex<float>));
    cudaMalloc((void**)&d_output, NUM_PIXELS * sizeof(float));
    
    // Copy data from host to device
    cudaMemcpy(d_pixels, h_pixels_flat.data(), h_pixels_flat.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_analytic, h_x_analytic_flat.data(), total_size * sizeof(thrust::complex<float>), cudaMemcpyHostToDevice);

    // // Constants
    float c = 344.351;
    // float Fs = 1.0e6;

    int blocksPerGrid = (NUM_PIXELS + threadsPerBlock - 1) / threadsPerBlock;
    beamforming_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_pixels, d_tx_coords, d_rx_coords, 
                                                           d_x_analytic, d_output, NUM_COORDS, 
                                                           c, FS, BINS);
    
    // // Host arrays for storing the results
    float *h_output;
    h_output = (float*)malloc(NUM_PIXELS * sizeof(float));
    if (h_output == NULL) {
        std::cerr << "Failed to allocate memory for h_output on the host." << std::endl;
        return -1;
    }

    // Copy output from device to host
    cudaMemcpy(h_output, d_output, NUM_PIXELS * sizeof(float), cudaMemcpyDeviceToHost);

    // Reshape the output into 150x150 grid on host
    float output_image[150][150];
    for (int i = 0; i < 150; i++) {
        for (int j = 0; j < 150; j++) {
            output_image[i][j] = h_output[i * 150 + j];
        }
    }

    // std::cout << "Output image generated successfully!" << std::endl;

    // // // Save the output image to a PNG file
    // // saveOutputImageToPNG("output_image.png", output_image);
    // // save as text file
    std::string filename = "/home/omkar/Desktop/Projetcs/Dome_experiments/output_image.csv";
    
    //flatten the 2D array
    std::vector<double> flat_output_image;
    for (int i = 0; i < 150; i++) {
        for (int j = 0; j < 150; j++) {
            flat_output_image.push_back(output_image[i][j]);
        }
    }

    //print length of flat_output_image
    std::cout << "Length of flat_output_image: " << flat_output_image.size() << std::endl;

    // Create and open the CSV file
    std::ofstream file(filename);

    // Check if the file opened successfully
    if (file.is_open()) {
        // Loop through the flattened vector and write to the file
        for (size_t i = 0; i < flat_output_image.size(); i++) {
            file << flat_output_image[i];
            if ((i + 1) % 150 == 0) {
                // Add a newline after every 150 elements to form rows in the CSV
                file << "\n";
            } else {
                // Add a comma after each element except the last one in the row
                file << ",";
            }
        }
        // Close the file after writing
        file.close();
        std::cout << "Output image saved to " << filename << std::endl;
    } else {
        std::cerr << "Could not open file " << filename << " for writing." << std::endl;
    }

    free(h_output);

    // Free device memory
    cudaFree(d_pixels);
    cudaFree(d_tx_coords);
    cudaFree(d_rx_coords);
    cudaFree(d_x_analytic);
    cudaFree(d_output);
    
    // Free host memory and other finalization steps...
    
    return 0;
}
