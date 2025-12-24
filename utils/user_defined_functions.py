import os
import sys
from statsmodels.tsa.stattools import adfuller, acf, pacf
import numpy as np
from scipy import ndimage
import scipy.signal as signal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
import math
from scipy.stats import pearsonr

## Calculate ARIMA parameters based on ACF and PACF plots
def calculate_arima_params(x, nlag = 50):
    lag_acf, confint_acf = acf(x[0:5000, 0], nlags=nlag, alpha=0.05)
    lag_pacf, confint_pacf = pacf(x[0:5000, 0], nlags=nlag, alpha=0.05)

    # Calculate the error bars (difference between value and confidence limit)
    # The "blue region" boundary is roughly 1.96 / sqrt(N) for large N
    lower_conf_acf = confint_acf[:, 0] - lag_acf
    upper_conf_acf = confint_acf[:, 1] - lag_acf
    def find_cutoff(values, conf_int):
        # confidence interval is centered at values, so we check if 0 is within [lower, upper]
        # conf_int[:, 0] is lower bound, conf_int[:, 1] is upper bound
        significant_lags = []
        for i in range(1, len(values)): # Start at 1 because lag 0 is always correlation 1
            if (0 < conf_int[i, 0]) or (0 > conf_int[i, 1]):
                significant_lags.append(i)
            else:
                # Once we hit the first insignificant lag, we stop (conservative approach)
                break
        return significant_lags[-1] if significant_lags else 0

    suggested_q = find_cutoff(lag_acf, confint_acf)
    suggested_p = find_cutoff(lag_pacf, confint_pacf)

    print(f"Visual inspection suggests: p (AR) = {suggested_p}, q (MA) = {suggested_q}")


## Calculate auto/cross-correlation matrix with delay
def correlation_calculation(x, y, range_min = 0, range_max = 50, step = 1):
    delay_range = range(range_min, range_max, step)
    cross_corr =[]
    for i in range(0, len(delay_range), 1):
        x_i = x[0:min(len(x),len(y))]
        y_i = ndimage.shift(x[0:min(len(x),len(y))], delay_range[i])
        cross_corr.append(np.corrcoef(x_i,y_i))
    return delay_range, cross_corr


def calculate_arima_params_downsampled(x, nlag = 50):
    lag_acf, confint_acf = acf(x, nlags=nlag, alpha=0.05)
    lag_pacf, confint_pacf = pacf(x, nlags=nlag, alpha=0.05)

    # Calculate the error bars (difference between value and confidence limit)
    # The "blue region" boundary is roughly 1.96 / sqrt(N) for large N
    lower_conf_acf = confint_acf[:, 0] - lag_acf
    upper_conf_acf = confint_acf[:, 1] - lag_acf
    def find_cutoff(values, conf_int):
        # confidence interval is centered at values, so we check if 0 is within [lower, upper]
        # conf_int[:, 0] is lower bound, conf_int[:, 1] is upper bound
        significant_lags = []
        for i in range(1, len(values)): # Start at 1 because lag 0 is always correlation 1
            if (0 < conf_int[i, 0]) or (0 > conf_int[i, 1]):
                significant_lags.append(i)
            else:
                # Once we hit the first insignificant lag, we stop (conservative approach)
                break
        return significant_lags[-1] if significant_lags else 0

    suggested_q = find_cutoff(lag_acf, confint_acf)
    suggested_p = find_cutoff(lag_pacf, confint_pacf)

    print(f"Visual inspection suggests: p (AR) = {suggested_p}, q (MA) = {suggested_q}")


def optimal_wiener_denoising_with_reference(full_signal, noise_reference, sampling_rate=2000):
    """
    Denoises the full signal using a specific noise-only reference segment.
    """
    # 1. Estimate the TRUE Noise PSD from the air-cutting segment
    # Lecture 3: PSD specifies how power is distributed over frequencies [cite: 1017]
    freqs, psd_noise = signal.welch(noise_reference, fs=sampling_rate, nperseg=1024)
    
    # 2. Estimate the PSD of the Full Noisy Signal
    _, psd_noisy = signal.welch(full_signal, fs=sampling_rate, nperseg=1024)
    
    # 3. Estimate Signal PSD (Subtractive method)
    # y(t) = x(t) + v(t) -> Phi_y = Phi_x + Phi_v [cite: 4, 15]
    psd_signal = np.maximum(psd_noisy - psd_noise, 0)
    
    # 4. Design the Transfer Function H(w)
    # H(w) is the ratio of signal power to total power
    H_w = psd_signal / (psd_signal + psd_noise)
    
    # 5. Apply the filter to the WHOLE measurement in the frequency domain
    Y_f = np.fft.rfft(full_signal)
    f_axis = np.fft.rfftfreq(len(full_signal), 1/sampling_rate)
    
    # Interpolate the H(w) weights to match the FFT frequency resolution
    H_interpolated = np.interp(f_axis, freqs, H_w)
    
    # Filter: X_hat(w) = Y(w) * H(w)
    X_hat_f = Y_f * H_interpolated
    
    return np.fft.irfft(X_hat_f, n=len(full_signal))


def evaluate_denoising_performance(raw_signal, clean_signal, noise_ref, sampling_rate=2000):
    """
    Evaluates filtering performance using SNR and Spectral Analysis.
    """
    
    # 1. Calculate Signal Power and Noise Power
    # We use the variance as a measure of power for WSS processes [cite: 901, 1015]
    power_raw = np.var(raw_signal)
    power_clean = np.var(clean_signal)
    power_noise_floor = np.var(noise_ref)

    # 2. SNR Estimation (Simplified for practical datasets)
    # SNR = 10 * log10(Power_signal / Power_noise)
    # Before filtering:
    snr_before = 10 * np.log10(power_raw / power_noise_floor)
    
    # After filtering: The noise floor in the clean signal should be lower.
    # We estimate residual noise by looking at the filtered noise_ref.
    # (Note: In your project, apply the same filter to the noise_ref to get this)
    # snr_after = 10 * np.log10(power_clean / residual_noise_power)
    
    print(f"--- Performance Metrics ---")
    print(f"Raw Signal Power: {power_raw:.4f}")
    print(f"Cleaned Signal Power: {power_clean:.4f}")
    print(f"Estimated SNR Improvement: {snr_before:.2f} dB (Initial)")

    # 3. Spectral Comparison (Lecture 3 & 7 Application)
    # This ensures we didn't lose the "treasure" frequencies (spindle multiples) [cite: 319, 320, 1017]
    f_raw, psd_raw = signal.welch(raw_signal, fs=sampling_rate, nperseg=1024)
    f_clean, psd_clean = signal.welch(clean_signal, fs=sampling_rate, nperseg=1024)
    
    plt.figure(figsize=(12, 6))
    plt.semilogy(f_raw, psd_raw, label='Raw Signal (Cutting)', alpha=0.5)
    plt.semilogy(f_clean, psd_clean, label='Filtered Signal (Wiener)', linewidth=2)
    plt.title("Power Spectral Density: Before vs After Filtering")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [V^2/Hz]")
    plt.grid(True, which="both", ls="-")
    
    # Highlight the spindle speed multiples mentioned in the report (e.g., if speed is 250Hz)
    # 
    for i in range(1, 5):
        plt.axvline(200 * i, color='r', linestyle='--', alpha=0.3, label='Spindle Harmonic' if i==1 else "")
    
    plt.legend()
    plt.show()


def plot_time_domain_denoising(raw_signal, clean_signal, sampling_rate=2000):
    """
    Plots the original, cleaned, and removed noise (residual) in the time domain.
    """
    time_axis = np.linspace(0, len(raw_signal) / sampling_rate, len(raw_signal))
    
    # Calculate the 'removed noise' (the residual)
    removed_noise = raw_signal - clean_signal
    
    plt.figure(figsize=(14, 10))

    # Plot 1: Original vs Cleaned
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, raw_signal, label='Raw Noisy Signal', color='red', alpha=0.7)
    plt.plot(time_axis, clean_signal, label='Cleaned Signal (Wiener)', color='blue', linewidth=1)
    plt.title("Time Domain: Raw vs. Cleaned Signal")
    plt.ylabel("Acceleration")
    plt.legend()

    # Plot 2: Zoomed view of a cutting segment
    # (Adjust indices to look at a specific tool impact)
    plt.subplot(3, 1, 2)
    start, end = int(10*sampling_rate), int(10.5*sampling_rate) # 0.5 sec window
    plt.plot(time_axis[start:end], raw_signal[start:end], color='red')
    plt.plot(time_axis[start:end], clean_signal[start:end], color='blue')
    plt.title("Zoomed View: Signal Recovery")
    plt.ylabel("Acceleration")

    # Plot 3: The 'Subtracted' Noise (Residual)
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, removed_noise, color='red', linewidth=0.5)
    plt.title("The 'Subtracted' Noise (What the filter removed)")
    plt.xlabel("Time [sec]")
    plt.ylabel("Noise Amplitude")
    
    plt.tight_layout()
    plt.show()


def fit_arma_44(signal_data, p, q):
    """
    Fits an ARMA(4,4) model to the provided signal.
    """
    # 1. Define the model order (p=4, d=0 for stationary, q=4)
    # [cite_start]Lecture 3: p is the AR order, q is the MA order [cite: 1006, 1008]
    model_order = (p, 0, q) 
    
    # 2. Initialize and fit the model using Maximum Likelihood
    # [cite_start]Lecture 6: ML estimators are asymptotically unbiased and efficient [cite: 48, 49]
    model = ARIMA(signal_data, order=model_order)
    model_fit = model.fit()
    
    # 3. Print the estimated coefficients (a1...a4 and b1...b4)
    print(model_fit.summary())
    
    return model_fit



def analyze_axis_orthogonality(x_axis_data, y_axis_data, sampling_rate=1000):
    # 1. Normalize the signals (Zero mean, Unit variance) 
    # This helps in seeing if correlation is 'around 0' regardless of signal scale
    x_norm = (x_axis_data - np.mean(x_axis_data)) / np.std(x_axis_data)
    y_norm = (y_axis_data - np.mean(y_axis_data)) / np.std(y_axis_data)
    
    # 2. Compute Cross-Correlation R_xy(tau)
    # 'full' mode gives lags from -(N-1) to +(N-1)
    r_xy = signal.correlate(x_norm, y_norm, mode='full') / len(x_norm)
    lags = signal.correlation_lags(len(x_norm), len(y_norm))
    
    # 3. Plotting the result
    plt.figure(figsize=(10, 5))
    plt.plot(lags, r_xy)
    plt.axhline(0, color='red', linestyle='--') # The Orthogonality line
    plt.title("Cross-Correlation R_xy (X-axis vs Y-axis)")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Correlation Coefficient")
    plt.grid(True)
    plt.show()

def resample_milling_data(data, original_fs=2000, target_fs=1000):
    """
    Resamples the data from original_fs to target_fs.
    """
    # 1. Calculate the number of samples in the new signal
    # New Length = (Original Length * Target Frequency) / Original Frequency
    number_of_samples = int(len(data) * target_fs / original_fs)
    
    # 2. Resample the signal
    # scipy.signal.resample uses an FFT-based method which inherently 
    # handles anti-aliasing in the frequency domain.
    resampled_data = signal.resample(data, number_of_samples)
    
    return resampled_data