#include "_signalprocessing.h"

std::vector<double> correctGroupDelay(const std::vector<double>& wfm, double gd, double fs) {
    size_t num_samples = wfm.size();
    double df = fs / num_samples;

    // Frequency index and frequency arrays combined into one loop
    std::vector<double> f(num_samples);
    std::vector<std::complex<double>> pr(num_samples);
    
    // Time delay (tau)
    double tau = gd / fs;

    for (size_t i = 0; i < num_samples; ++i) {
        f[i] = i * df;
        if (f[i] > (fs / 2)) {
            f[i] -= fs;
        }
        std::complex<double> complex_phase(0, tau * 2 * M_PI * f[i]);
        pr[i] = std::exp(complex_phase);
    }

    // FFTW setup
    fftw_complex *in, *out;
    fftw_plan forward_plan, backward_plan;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * num_samples);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * num_samples);

    if (!in || !out) {
        std::cerr << "Error: FFTW memory allocation failed." << std::endl;
        if (in) fftw_free(in);
        if (out) fftw_free(out);
        return std::vector<double>();
    }

    // Fill the input array for FFTW
    for (size_t j = 0; j < num_samples; ++j) {
        in[j][0] = wfm[j];  // Real part
        in[j][1] = 0.0;     // Imaginary part
    }

    // Perform FFT
    forward_plan = fftw_plan_dft_1d(num_samples, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(forward_plan);

    // Apply the phase correction
    for (size_t j = 0; j < num_samples; ++j) {
        std::complex<double> out_complex(out[j][0], out[j][1]);
        std::complex<double> corrected_complex = out_complex * pr[j];
        out[j][0] = corrected_complex.real();
        out[j][1] = corrected_complex.imag();
    }

    // Perform IFFT
    backward_plan = fftw_plan_dft_1d(num_samples, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(backward_plan);

    // Normalize and store the result
    std::vector<double> wfm_correct(num_samples);
    for (size_t j = 0; j < num_samples; ++j) {
        wfm_correct[j] = in[j][0] / num_samples;  // Only real part
    }

    // Destroy FFTW plans and free memory
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_free(in);
    fftw_free(out);

    return wfm_correct;
}

std::vector<double> generate_lfm_signal(double f_start, double f_stop, double fs, double sweep_length, double amp) {
    // Calculate the number of samples
    int num_samples = static_cast<int>(fs * sweep_length);
    
    // Time vector
    std::vector<double> t(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        t[i] = i / fs;
    }

    // Frequency sweep rate (chirp rate)
    double k = (f_stop - f_start) / sweep_length;

    // Generate LFM signal
    std::vector<double> s(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        s[i] = amp * std::cos(2 * M_PI * (f_start * t[i] + 0.5 * k * t[i] * t[i]));
    }
    return s;
}

// Hilbert Transform using FFT
std::vector<std::complex<double>> hilbert_transform(const std::vector<double>& signal) {
    int n = signal.size();
    std::vector<std::complex<double>> analytic_signal(n);

    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

    // Copy the real signal to the input
    for (int i = 0; i < n; ++i) {
        in[i][0] = signal[i]; // Real part
        in[i][1] = 0.0;       // Imaginary part
    }

    // Perform FFT
    fftw_plan plan = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Multiply by the appropriate frequency response for the Hilbert transform
    for (int i = 0; i < n; ++i) {
        if (i == 0 || (i == n / 2 && n % 2 == 0)) {
            out[i][0] = out[i][0]; // DC and Nyquist frequencies unchanged
            out[i][1] = out[i][1];
        } else if (i < n / 2) {
            out[i][0] *= 2; // Positive frequencies are multiplied by 2
            out[i][1] *= 2;
        } else {
            out[i][0] = 0; // Negative frequencies are zeroed out
            out[i][1] = 0;
        }
    }

    // Perform inverse FFT
    fftw_plan ifft_plan = fftw_plan_dft_1d(n, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(ifft_plan);

    // Normalize and copy to the output
    for (int i = 0; i < n; ++i) {
        analytic_signal[i] = std::complex<double>(in[i][0] / n, in[i][1] / n);
    }

    fftw_destroy_plan(plan);
    fftw_destroy_plan(ifft_plan);
    fftw_free(in);
    fftw_free(out);

    return analytic_signal;
}

std::vector<std::complex<double>> fft(const std::vector<std::complex<double>>& signal) {
    int n = signal.size();
    std::vector<std::complex<double>> result(n);

    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

    for (int i = 0; i < n; ++i) {
        in[i][0] = signal[i].real();
        in[i][1] = signal[i].imag();
    }

    fftw_plan plan = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (int i = 0; i < n; ++i) {
        result[i] = std::complex<double>(out[i][0], out[i][1]);
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return result;
}

std::vector<std::complex<double>> ifft(const std::vector<std::complex<double>>& signal) {
    int n = signal.size();
    std::vector<std::complex<double>> result(n);

    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

    for (int i = 0; i < n; ++i) {
        in[i][0] = signal[i].real();
        in[i][1] = signal[i].imag();
    }

    fftw_plan plan = fftw_plan_dft_1d(n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (int i = 0; i < n; ++i) {
        result[i] = std::complex<double>(out[i][0] / n, out[i][1] / n);
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return result;
}

// Match filtering in frequency domain
std::vector<std::complex<double>> match_filter(const std::vector<std::complex<double>>& x_hilbert, const std::vector<double>& lfm_signal) {
    int n = x_hilbert.size();

    // Zero-pad LFM signal if necessary
    std::vector<std::complex<double>> lfm_signal_padded(n, 0.0);
    for (int i = 0; i < std::min(static_cast<int>(lfm_signal.size()), n); ++i) {
        lfm_signal_padded[i] = lfm_signal[i];
    }

    // FFT of the padded LFM signal
    std::vector<std::complex<double>> t_fft = fft(lfm_signal_padded);

    // Compute the inverse FFT of the product of FFTs (match filtering)
    std::vector<std::complex<double>> x_analytic_fft(n);
    for (int i = 0; i < n; ++i) {
        x_analytic_fft[i] = x_hilbert[i] * std::conj(t_fft[i]);
    }

    return ifft(x_analytic_fft);
}

std::vector<double> computeMeanSignal(const std::vector<std::vector<double>>& allData) {

    int array_size = allData[0].size(); 

    std::vector<double> meanSignal(array_size, 0.0); // Initialize mean signal with 999 zeros

    if (allData.empty()) {
        std::cerr << "No data to compute mean signal." << std::endl;
        return meanSignal;
    }

    // Compute the mean for each column
    for (size_t i = 0; i < array_size; ++i) {
        double sum = 0.0;
        for (const auto& fileData : allData) {
            sum += fileData[i];
        }
        meanSignal[i] = sum / allData.size();
    }

    return meanSignal;
}

std::vector<std::vector<double>> subtractMeanBackground(
    const std::vector<std::vector<double>>& signal,
    const std::vector<double>& meanBackground) {

    int array_size = meanBackground.size();

    std::vector<std::vector<double>> result(signal.size(), std::vector<double>(array_size, 0.0));

    for (size_t i = 0; i < signal.size(); ++i) {
        for (size_t j = 0; j < array_size; ++j) {
            result[i][j] = signal[i][j] - meanBackground[j];
        }
    }

    return result;
}