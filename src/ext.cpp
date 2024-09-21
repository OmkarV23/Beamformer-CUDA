// import pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <complex>

#include "_readsignal.h"
#include "_signalprocessing.h"
#include "beamformer.cuh"

namespace py = pybind11;

py::array_t<double> ReadSignals(const std::string& folder) {
    std::vector<std::vector<double>> signals = readAllCSVFiles(folder);

    size_t num_rows = signals.size();
    size_t num_cols = 0;

    if (num_rows > 0) {
        num_cols = signals[0].size();
        for (size_t i = 1; i < num_rows; ++i) {
            if (signals[i].size() != num_cols) {
                throw std::runtime_error("Inconsistent number of columns in signals data");
            }
        }
    } else {
        return py::array_t<double>();
    }

    py::array_t<double> result({num_rows, num_cols});

    py::buffer_info buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);

    for (size_t i = 0; i < num_rows; ++i) {
        std::memcpy(ptr + i * num_cols, signals[i].data(), num_cols * sizeof(double));
    }

    return result;
}

py::array_t<double> LFM(const double f_start, const double f_stop, const double fs, const double sweep_length, const double amp) {
    std::vector<double> lfm_signal = generate_lfm_signal(f_start, f_stop, fs, sweep_length, amp);

    size_t num_samples = lfm_signal.size();

    py::array_t<double> result({num_samples});

    py::buffer_info buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);

    std::memcpy(ptr, lfm_signal.data(), num_samples * sizeof(double));

    return result;
}

py::array_t<double> CorrectGroupDelay(const py::array_t<double>& wfm, const double gd, const double fs) {
    py::buffer_info buf = wfm.request();
    double* ptr = static_cast<double*>(buf.ptr);

    size_t num_samples = buf.size;

    std::vector<double> wfm_signal(num_samples);
    std::memcpy(wfm_signal.data(), ptr, num_samples * sizeof(double));

    std::vector<double> corrected_signal = correctGroupDelay(wfm_signal, gd, fs);

    py::array_t<double> result({num_samples});

    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    std::memcpy(ptr_result, corrected_signal.data(), num_samples * sizeof(double));

    return result;
}

py::array_t<std::complex<double>> HilbertTransform(const py::array_t<double>& signal) {
    py::buffer_info buf = signal.request();
    double* ptr = static_cast<double*>(buf.ptr);

    size_t num_samples = buf.size;

    std::vector<double> real_signal(num_samples);
    std::memcpy(real_signal.data(), ptr, num_samples * sizeof(double));

    std::vector<std::complex<double>> analytic_signal = hilbert_transform(real_signal);

    py::array_t<std::complex<double>> result({num_samples});

    py::buffer_info buf_result = result.request();
    std::complex<double>* ptr_result = static_cast<std::complex<double>*>(buf_result.ptr);

    std::memcpy(ptr_result, analytic_signal.data(), num_samples * sizeof(std::complex<double>));

    return result;
}

py::array_t<std::complex<double>> MatchFilter(const py::array_t<std::complex<double>>& x_hilbert, const py::array_t<double>& lfm_signal) {
    py::buffer_info buf_x_hilbert = x_hilbert.request();
    std::complex<double>* ptr_x_hilbert = static_cast<std::complex<double>*>(buf_x_hilbert.ptr);

    size_t num_samples = buf_x_hilbert.size;

    std::vector<std::complex<double>> x_hilbert_signal(num_samples);
    std::memcpy(x_hilbert_signal.data(), ptr_x_hilbert, num_samples * sizeof(std::complex<double>));

    py::buffer_info buf_lfm_signal = lfm_signal.request();
    double* ptr_lfm_signal = static_cast<double*>(buf_lfm_signal.ptr);

    size_t num_samples_lfm = buf_lfm_signal.size;

    std::vector<double> lfm_signal_vec(num_samples_lfm);
    std::memcpy(lfm_signal_vec.data(), ptr_lfm_signal, num_samples_lfm * sizeof(double));

    std::vector<std::complex<double>> matched_signal = match_filter(x_hilbert_signal, lfm_signal_vec);

    py::array_t<std::complex<double>> result({num_samples});

    py::buffer_info buf_result = result.request();
    std::complex<double>* ptr_result = static_cast<std::complex<double>*>(buf_result.ptr);

    std::memcpy(ptr_result, matched_signal.data(), num_samples * sizeof(std::complex<double>));

    return result;
}

py::array_t<double> CreatePixelSpace(double x_min, double x_max, double y_min, double y_max,
                                                     int num_points_per_side, double z_value)
{
    std::vector<std::vector<double>> Pixels_2D = createPixelSpace(x_min, x_max, y_min, y_max, num_points_per_side, z_value);

    size_t num_rows = Pixels_2D.size();
    size_t num_cols = 0;

    if (num_rows > 0) {
        num_cols = Pixels_2D[0].size();
        for (size_t i = 1; i < num_rows; ++i) {
            if (Pixels_2D[i].size() != num_cols) {
                throw std::runtime_error("Inconsistent number of columns in signals data");
            }
        }
    } else {
        return py::array_t<double>();
    }

    py::array_t<double> result({num_rows, num_cols});

    py::buffer_info buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);

    for (size_t i = 0; i < num_rows; ++i) {
        std::memcpy(ptr + i * num_cols, Pixels_2D[i].data(), num_cols * sizeof(double));
    }

    return result;

}

py::array_t<double> Generate_Coordinates(double radius, double z)
{
    std::vector<std::vector<double>> Sensor_pos = GenerateSensorCoordinates(radius, z);

    size_t num_rows = Sensor_pos.size();
    size_t num_cols = 0;

    if (num_rows > 0) {
        num_cols = Sensor_pos[0].size();
        for (size_t i = 1; i < num_rows; ++i) {
            if (Sensor_pos[i].size() != num_cols) {
                throw std::runtime_error("Inconsistent number of columns in signals data");
            }
        }
    } else {
        return py::array_t<double>();
    }

    py::array_t<double> result({num_rows, num_cols});

    py::buffer_info buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);

    for (size_t i = 0; i < num_rows; ++i) {
        std::memcpy(ptr + i * num_cols, Sensor_pos[i].data(), num_cols * sizeof(double));
    }

    return result;
}

py::array_t<std::complex<double>> Beamform(const py::array_t<double>& h_pixels_flat, const py::array_t<double>& h_tx_coords, const py::array_t<double>& h_rx_coords, 
                                           const py::array_t<std::complex<double>>& h_x_analytic_flat, const int num_pixels, const int num_coords, 
                                           const double c, const double Fs, const int signal_length) 
{
    py::buffer_info buf_h_pixels_flat = h_pixels_flat.request();
    double* ptr_h_pixels_flat = static_cast<double*>(buf_h_pixels_flat.ptr);

    size_t num_samples = buf_h_pixels_flat.size;

    std::vector<double> h_pixels_flat_vec(num_samples);
    std::memcpy(h_pixels_flat_vec.data(), ptr_h_pixels_flat, num_samples * sizeof(double));

    py::buffer_info buf_h_tx_coords = h_tx_coords.request();
    double* ptr_h_tx_coords = static_cast<double*>(buf_h_tx_coords.ptr);

    size_t num_samples_tx = buf_h_tx_coords.size;

    std::vector<double> h_tx_coords_vec(num_samples_tx);
    std::memcpy(h_tx_coords_vec.data(), ptr_h_tx_coords, num_samples_tx * sizeof(double));

    py::buffer_info buf_h_rx_coords = h_rx_coords.request();
    double* ptr_h_rx_coords = static_cast<double*>(buf_h_rx_coords.ptr);

    size_t num_samples_rx = buf_h_rx_coords.size;

    std::vector<double> h_rx_coords_vec(num_samples_rx);
    std::memcpy(h_rx_coords_vec.data(), ptr_h_rx_coords, num_samples_rx * sizeof(double));

    py::buffer_info buf_h_x_analytic_flat = h_x_analytic_flat.request();
    std::complex<double>* ptr_h_x_analytic_flat = static_cast<std::complex<double>*>(buf_h_x_analytic_flat.ptr);

    size_t num_samples_x = buf_h_x_analytic_flat.size;

    std::vector<std::complex<double>> h_x_analytic_flat_vec(num_samples_x);
    std::memcpy(h_x_analytic_flat_vec.data(), ptr_h_x_analytic_flat, num_samples_x * sizeof(std::complex<double>));

    std::vector<std::complex<double>> beamformed_signal = _beamform_extern(h_pixels_flat_vec, h_tx_coords_vec, h_rx_coords_vec, h_x_analytic_flat_vec, num_pixels, num_coords, c, Fs, signal_length);

    py::array_t<std::complex<double>> result({num_pixels});

    py::buffer_info buf_result = result.request();
    std::complex<double>* ptr_result = static_cast<std::complex<double>*>(buf_result.ptr);

    std::memcpy(ptr_result, beamformed_signal.data(), num_pixels * sizeof(std::complex<double>));

    return result;
}

PYBIND11_MODULE(FastBeamformer, m) {
    m.def("ReadSignals", &ReadSignals, "Read CSV files from a folder and return a NumPy array");
    m.def("LFM", &LFM, "Generate a Linear Frequency Modulated (LFM) signal and return a NumPy array");
    m.def("CorrectGroupDelay", &CorrectGroupDelay, "Correct group delay in a waveform signal and return a NumPy array");
    m.def("HilbertTransform", &HilbertTransform, "Compute the Hilbert transform of a real signal and return a NumPy array");
    m.def("MatchFilter", &MatchFilter, "Apply a matched filter to a signal and return a NumPy array");
    m.def("CreatePixelSpace", &CreatePixelSpace, "Create a 2D pixel space and return a NumPy array");
    m.def("Generate_Coordinates", &Generate_Coordinates, "Generate sensor coordinates and return a NumPy array");
    m.def("Beamform", &Beamform, "Apply the Fast Beamformer algorithm and return a NumPy array");
}


