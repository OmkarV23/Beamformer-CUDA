#ifndef SIGNAL_PROCESSING_H
#define SIGNAL_PROCESSING_H

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fftw3.h>
#include <algorithm>

std::vector<double> correctGroupDelay(const std::vector<double>& wfm, 
                                        double gd, 
                                        double fs);

std::vector<double> generate_lfm_signal(double f_start, 
                                        double f_stop, 
                                        double fs, 
                                        double sweep_length,
                                        double amp);

std::vector<std::complex<double>> hilbert_transform(const std::vector<double>& signal);

std::vector<std::complex<double>> fft(const std::vector<std::complex<double>>& signal);

std::vector<std::complex<double>> ifft(const std::vector<std::complex<double>>& signal);

std::vector<std::complex<double>> match_filter(const std::vector<std::complex<double>>& x_hilbert, 
                                                const std::vector<double>& lfm_signal);

std::vector<double> computeMeanSignal(const std::vector<std::vector<double>>& allData);

std::vector<std::vector<double>> subtractMeanBackground(
                                        const std::vector<std::vector<double>>& signal,
                                        const std::vector<double>& meanBackground);

#endif // SIGNAL_PROCESSING_H