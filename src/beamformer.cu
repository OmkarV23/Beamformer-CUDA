#include "beamformer.cuh"


__global__ void beamforming_kernel(double *pixels, double *tx_coords, double *rx_coords, 
                                   thrust::complex<double> *x_analytic, thrust::complex<double> *output, 
                                   int num_coords, double c, double Fs, int signal_length) {
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pixel_idx < 22500) {  // 150x150 = 22500 pixels
        double d1, d2, tof;
        thrust::complex<double> sum(0.0, 0.0);
        
        for (int i = 0; i < num_coords; i++) {
            d1 = sqrt(pow(pixels[3 * pixel_idx] - tx_coords[3 * i], 2) + 
                       pow(pixels[3 * pixel_idx + 1] - tx_coords[3 * i + 1], 2) + 
                       pow(pixels[3 * pixel_idx + 2] - tx_coords[3 * i + 2], 2));
            
            d2 = sqrt(pow(pixels[3 * pixel_idx] - rx_coords[3 * i], 2) + 
                       pow(pixels[3 * pixel_idx + 1] - rx_coords[3 * i + 1], 2) + 
                       pow(pixels[3 * pixel_idx + 2] - rx_coords[3 * i + 2], 2));

            tof = ((d1 + d2) / c) * Fs;

            // Sinc interpolation and summing values at the pixels
            for (int j = 0; j < signal_length; j++) {
                double sinc_value = (j == tof) ? 1.0f : sin(M_PI * (tof-j)) / (M_PI * (tof-j) + 1e-8);
                sum += x_analytic[i * signal_length + j] * sinc_value;
            }
        }
        output[pixel_idx] = sum;
    }
}


__global__ void generate_circle_coordinates_kernel(double *tx_coords, double *rx_coords, 
                                                   double radius_tx, double radius_rx, 
                                                   double z_tx, double z_rx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < 360) {  // Loop over all angles (degrees) from 1 to 360
        double theta = (i + 1) * (M_PI / 180.0);  // Convert degrees to radians

        // Coordinates for the circle in XY plane (X, Y)
        double x_tx = radius_tx * cos(theta);
        double y_tx = radius_tx * sin(theta);

        double x_rx = radius_rx * cos(theta);
        double y_rx = radius_rx * sin(theta);

        // Set the coordinates in the tx_coords and rx_coords arrays
        tx_coords[i * 3] = x_tx;
        tx_coords[i * 3 + 1] = y_tx;
        tx_coords[i * 3 + 2] = z_tx;

        rx_coords[i * 3] = x_rx;
        rx_coords[i * 3 + 1] = y_rx;
        rx_coords[i * 3 + 2] = z_rx;
    }
}


std::vector<std::vector<double>> createPixelSpace(double x_min, double x_max, double y_min, double y_max,
                                                     int num_points_per_side, double z_value) {
    // Calculate the step size for x and y directions
    double x_step = (x_max - x_min) / (num_points_per_side - 1);
    double y_step = (y_max - y_min) / (num_points_per_side - 1);

    // Initialize the pixel space with size (22500, 3)
    std::vector<std::vector<double>> pixel_space(num_points_per_side * num_points_per_side, std::vector<double>(3, 0.0f));

    // Fill the pixel space
    int index = 0;
    for (int i = 0; i < num_points_per_side; ++i) {
        for (int j = 0; j < num_points_per_side; ++j) {
            double x = x_min + j * x_step;
            double y = y_min + i * y_step;
            pixel_space[index][0] = x;
            pixel_space[index][1] = y;
            pixel_space[index][2] = z_value;
            ++index;
        }
    }

    return pixel_space;
}

std::vector<std::vector<double>> GenerateSensorCoordinates(double radius, double z) {
    std::vector<std::vector<double>> sensor_coordinates;
    for (int i = 0; i < 360; i++) {
        double theta = (i + 1) * (M_PI / 180.0);
        double x = radius * cos(theta);
        double y = radius * sin(theta);
        sensor_coordinates.push_back({x, y, z});
    }
    return sensor_coordinates;
}

std::vector<std::complex<double>> _beamform_extern(std::vector<double> h_pixels_flat, std::vector<double> h_tx_coords, std::vector<double> h_rx_coords, 
                                            std::vector<std::complex<double>> h_x_analytic_flat, int num_pixels,
                                            int num_coords, double c, double Fs, int signal_length) 
{
    size_t total_size = h_x_analytic_flat.size();

    std::vector<std::complex<double>> h_output(num_pixels);
    for (int i = 0; i < num_pixels; i++) {
        h_output[i] = std::complex<double>(0.0, 0.0);
    }

    double *d_tx_coords, *d_rx_coords;

    double *d_pixels;
    thrust::complex<double>* d_output;
    thrust::complex<double>* d_x_analytic;

    cudaMalloc((void**)&d_pixels, h_pixels_flat.size() * sizeof(double));
    cudaMalloc((void**)&d_x_analytic, total_size * sizeof(thrust::complex<double>));
    cudaMalloc((void**)&d_output, num_pixels * sizeof(thrust::complex<double>));

    cudaMalloc((void**)&d_tx_coords, NUM_COORDS * sizeof(TX_INITIAL) * sizeof(double));
    cudaMalloc((void**)&d_rx_coords, NUM_COORDS * sizeof(RX_INITIAL) * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_pixels, h_pixels_flat.data(), h_pixels_flat.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output.data(), num_pixels * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_analytic, h_x_analytic_flat.data(), total_size * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);

    cudaMemcpy(d_tx_coords, h_tx_coords.data(), NUM_COORDS * sizeof(TX_INITIAL) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rx_coords, h_rx_coords.data(), NUM_COORDS * sizeof(RX_INITIAL) * sizeof(double), cudaMemcpyHostToDevice);


    int blocksPerGrid = (num_pixels + 256 - 1) / 256;
    beamforming_kernel<<<blocksPerGrid, 256>>>(d_pixels, d_tx_coords, d_rx_coords, 
                                               d_x_analytic, d_output, num_coords, c, Fs, signal_length);

    cudaMemcpy(h_output.data(), d_output, num_pixels * sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);
    cudaFree(d_tx_coords);
    cudaFree(d_rx_coords);
    cudaFree(d_x_analytic);
    cudaFree(d_output);

    return h_output;

}

int main() {

    std::vector<std::vector<double>> background = readAllCSVFiles(background_dir);
    std::vector<double> mean_background = computeMeanSignal(background);


    std::vector<std::vector<double>> signal = readAllCSVFiles(signal_dir);

    std::vector<std::vector<double>> signal_minus_background = subtractMeanBackground(signal, mean_background);

    // std::vector<std::vector<double>> signal_minus_background = readAllCSVFiles(signal_dir);

    std::vector<double> LFM = generate_lfm_signal(F_START, F_STOP, FS, SWEEP_LENGTH, AMP);


    std::vector<thrust::complex<double>> h_x_analytic_flat;
    for (int i = 0; i < signal_minus_background.size(); i++) {
        std::vector<std::complex<double>> x_ana = match_filter(hilbert_transform(correctGroupDelay(signal_minus_background[i], GRP_DELAY, FS)), LFM);
        for (int j = 0; j < x_ana.size(); j++) {
            h_x_analytic_flat.emplace_back(static_cast<double>(x_ana[j].real()), static_cast<double>(x_ana[j].imag()));
        }
    }

    // The total size would be num_coords * bins (flattened size)
    size_t total_size = h_x_analytic_flat.size();

    // Create the pixel space
    std::vector<std::vector<double>> h_pixels = createPixelSpace(x_min, x_max, y_min, y_max, num_points_per_side, z_value);

    std::vector<double> h_pixels_flat;
    for (int i = 0; i < h_pixels.size(); ++i) {
        h_pixels_flat.push_back(h_pixels[i][0]);
        h_pixels_flat.push_back(h_pixels[i][1]);
        h_pixels_flat.push_back(h_pixels[i][2]);
    }


    double *d_tx_coords, *d_rx_coords;
    cudaMalloc((void**)&d_tx_coords, NUM_COORDS * sizeof(TX_INITIAL) * sizeof(double));
    cudaMalloc((void**)&d_rx_coords, NUM_COORDS * sizeof(RX_INITIAL) * sizeof(double));


    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocks1 = (360 + threadsPerBlock - 1) / threadsPerBlock;
    generate_circle_coordinates_kernel<<<blocks1, threadsPerBlock>>>(d_tx_coords, d_rx_coords, 
                                                                            RADIUS_TX, RADIUS_RX, TX_INITIAL[2], RX_INITIAL[2]);


    std::vector<std::complex<double>> h_output(NUM_PIXELS);
    for (int i = 0; i < NUM_PIXELS; i++) {
        h_output[i] = std::complex<double>(0.0, 0.0);
    }

    // Allocate memory on the GPU (Device)
    double *d_pixels;
    thrust::complex<double>* d_output;
    thrust::complex<double>* d_x_analytic;

    cudaMalloc((void**)&d_pixels, h_pixels_flat.size() * sizeof(double));
    cudaMalloc((void**)&d_x_analytic, total_size * sizeof(thrust::complex<double>));
    cudaMalloc((void**)&d_output, NUM_PIXELS * sizeof(thrust::complex<double>));
    
    // Copy data from host to device
    cudaMemcpy(d_pixels, h_pixels_flat.data(), h_pixels_flat.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output.data(), NUM_PIXELS * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_analytic, h_x_analytic_flat.data(), total_size * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);


    // Timing variables
    cudaEvent_t start, stop;
    float milliseconds = 0;

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording time
    cudaEventRecord(start);
    

    int blocksPerGrid = (NUM_PIXELS + 256 - 1) / 256;
    beamforming_kernel<<<blocksPerGrid, 256>>>(d_pixels, d_tx_coords, d_rx_coords, 
                                                           d_x_analytic, d_output, NUM_COORDS, 
                                                           SPEED_OF_SOUND, FS, BINS);

    // Stop recording time
    cudaEventRecord(stop);

    // Synchronize the events to wait for the kernel to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output the execution time
    std::cout << "Beamforming kernel execution time: " << milliseconds << " ms" << std::endl;

    // Cleanup CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_output.data(), d_output, NUM_PIXELS * sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost);


    std::ofstream
    file(image_file);

    // Check if the file opened successfully
    if (file.is_open()) {
        // Loop through the flattened vector and write to the file
        for (size_t i = 0; i < NUM_PIXELS; i++) {
            file << h_output[i].real() << "," << h_output[i].imag() << "\n";
        }
        // Close the file after writing
        file.close();
        std::cout << "Output image saved to " << image_file << std::endl;
    } else {
        std::cerr << "Could not open file " << image_file << " for writing." << std::endl;
    }

    cudaFree(d_pixels);
    cudaFree(d_tx_coords);
    cudaFree(d_rx_coords);
    cudaFree(d_x_analytic);
    cudaFree(d_output);
        
    return 0;
}
