import numpy as np
import mne
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from contextlib import redirect_stdout
import os
from functools import wraps

def silenced(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with open(os.devnull, 'w') as f:
            with redirect_stdout(f):
                return func(*args, **kwargs)
    return wrapper

# Or wrap an existing function on the fly
# clean_preprocess = silenced(pre_process)

def zero_mean(obj):
    """
    Applies zero-mean normalization across timesteps for each electrode.
    Expects input shape: (electrodes, timesteps)
    """
    mean_per_channel = np.mean(obj.get_data(), axis=1, keepdims=True)
    zero_meaned = obj.get_data() - mean_per_channel
    return mne.io.RawArray(zero_meaned, obj.info)

def downsample(raw_obj, factor):
    """
    Downsamples the MNE Raw object by a given factor based on the current sampling rate.
    """
    current_sfreq = raw_obj.info['sfreq']
    new_sfreq = current_sfreq / factor
    print(f"--- Downsampling from {current_sfreq} Hz to {new_sfreq} Hz ---")
    raw_obj.resample(sfreq=new_sfreq)
    return raw_obj

def normalization(scan_matrix, technique="minmax"):
    """
    Applies MinMax or Z-score normalization across timesteps for each electrode.
    Expects input shape: (electrodes, timesteps)
    """
    print(f"--- Applying {technique} normalization ---")
    # Sklearn expects (samples, features), so we transpose to normalize across time per channel
    matrix_T = scan_matrix.T 
    
    if technique.lower() == "minmax":
        scaler = MinMaxScaler(feature_range=(-1, 1))
        normalized_T = scaler.fit_transform(matrix_T)
    elif technique.lower() == "zscore":
        scaler = StandardScaler()
        normalized_T = scaler.fit_transform(matrix_T)
    else:
        raise ValueError("Technique must be 'minmax' or 'zscore'")
        
    return normalized_T.T

def normalization_3d(wavelet_output, technique="minmax"):
    """
    Normalize each electrode-band time series independently.

    Input shape: (electrodes, bands, timesteps).
    For `minmax`: scales each (electrode, band) time series to [-1, 1].
    For `zscore`: standardizes each (electrode, band) time series (mean=0, std=1).
    Returns array with the same shape.
    """

    data = wavelet_output.astype(float)

    if technique.lower() == "minmax":
        mins = np.min(data, axis=2, keepdims=True)
        maxs = np.max(data, axis=2, keepdims=True)
        denom = maxs - mins
        denom[denom == 0] = 1.0
        scaled = (data - mins) / denom  # 0..1
        normalized_3d = (scaled * 2.0) - 1.0  # -1..1

    elif technique.lower() == "zscore":
        means = np.mean(data, axis=2, keepdims=True)
        stds = np.std(data, axis=2, keepdims=True)
        stds[stds == 0] = 1.0
        normalized_3d = (data - means) / stds

    else:
        raise ValueError("Technique must be 'minmax' or 'zscore'")

    return normalized_3d
    

def noise_reduction(raw_obj):
    """
    Applies statistical artifact reduction, bandpass filtering, 
    and ICA to clear eye blinks and muscle artifacts without needing geometry.
    """
    print("--- Starting Noise Reduction ---")
    
    # 1. Bandpass filter to suppress slow drifts (<0.5Hz) and high-frequency muscle noise (>100Hz)
    raw_obj.filter(l_freq=0.5, h_freq=100.0, fir_design='firwin')
    
    # 2. Automated Bad Channel Detection (Alternative to Maxwell physics)
    #print("Detecting malfunctioning sensors via amplitude variance...")
    
    # Extract data matrix to calculate standard deviation across channels
    data = raw_obj.get_data()
    stds = np.std(data, axis=1)
    
    # Let's find channels that are completely dead (flat) or have massive outliers (malfunctioning)
    # Since raw data scaling can vary, we use standard median-based thresholds
    median_std = np.median(stds)
    
    # Thresholds: flat if std is close to 0, bad if std is 10x higher than median channel
    flat_channels_idx = np.where(stds < 1e-20)[0]
    noisy_channels_idx = np.where(stds > (median_std * 10))[0]
    
    bad_indices = list(flat_channels_idx) + list(noisy_channels_idx)
    
    # Map back to channel names
    bad_channels = [raw_obj.ch_names[idx] for idx in bad_indices]
    
    raw_obj.info['bads'] = bad_channels
    #print(f"Interpolating bad channels: {raw_obj.info['bads']}")
    
    # Since we don't have spatial coordinates to interpolate, we drop them or zero them out.
    # To keep your matrix size exactly 248 channels, we just clear the bad channels' content.
    if len(bad_channels) > 0:
        for ch in bad_channels:
            ch_idx = raw_obj.ch_names.index(ch)
            raw_obj._data[ch_idx, :] = 0.0 # Clear out noisy channels so they don't break the Fourier Transform
    
    # 3. Independent Component Analysis (ICA) for Eye Blinks and Body Movements
    #print("Running ICA for ocular and motor artifact rejection...")
    
    # Removed 'tol' to fix the TypeError, kept max_iter high for your MacBook Air
    ica = mne.preprocessing.ICA(n_components=0.95, random_state=42, method='fastica', max_iter=1000)
    ica.fit(raw_obj)
    
    #print("Automatically identifying artifact components using statistical heuristics...")
    
    # Extract the independent component source time-series
    ica_sources = ica.get_sources(raw_obj).get_data() # shape: (components, timesteps)
    
    # --- HEURISTIC 1: KCC (Kurtosis) to detect Eye Blinks ---
    from scipy.stats import kurtosis
    kurt_scores = kurtosis(ica_sources, axis=1)
    kurt_threshold = np.median(kurt_scores) + (3 * np.std(kurt_scores))
    blink_idx = np.where(kurt_scores > kurt_threshold)[0]
    
    # --- HEURISTIC 2: High-Frequency Variance to detect Muscle Bursts ---
    fft_vals = np.abs(np.fft.rfft(ica_sources, axis=1))
    freqs = np.fft.rfftfreq(ica_sources.shape[1], d=1.0/raw_obj.info['sfreq'])
    
    high_freq_mask = freqs > 40.0
    muscle_scores = np.mean(fft_vals[:, high_freq_mask], axis=1)
    muscle_threshold = np.median(muscle_scores) + (3 * np.std(muscle_scores))
    muscle_idx = np.where(muscle_scores > muscle_threshold)[0]
    
    # Combine both lists of bad components safely
    bad_components = list(set(list(blink_idx) + list(muscle_idx)))
    
    ica.exclude = bad_components
    #print(f"Excluding {len(bad_components)} ICA components (Blinks: {len(blink_idx)}, Muscle: {len(muscle_idx)}).")
    
    # Apply the ICA cleaning to the data
    raw_obj = ica.apply(raw_obj)
    return raw_obj

def aggregate_bands(frequency_data, frequencies, data_type="fourier"):
    """
    Aggregates continuous frequency data into distinct brain wave bands.
    
    Parameters:
    - frequency_data: 
        If data_type="fourier": 2D array of shape (#electrodes, #frequencies)
        If data_type="wavelets": 3D array of shape (#electrodes, #frequencies, #timesteps)
    - frequencies: 1D array containing the frequency values in Hz
    - data_type: "fourier" or "wavelets"
    """
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 100)
    }
    
    aggregated_list = []
    
    for band_name, (fmin, fmax) in bands.items():
        freq_idx = np.logical_and(frequencies >= fmin, frequencies < fmax)
        
        if data_type.lower() == "fourier":
            band_power = frequency_data[:, freq_idx].mean(axis=1)
            aggregated_list.append(band_power)
            
        elif data_type.lower() == "wavelets":
            band_power_over_time = frequency_data[:, freq_idx, :].mean(axis=1)
            aggregated_list.append(band_power_over_time)
            
    if data_type.lower() == "fourier":
        return np.array(aggregated_list).T
        
    elif data_type.lower() == "wavelets":
        return np.array(aggregated_list).transpose(1, 0, 2)

@silenced
def pre_process(scan, sfreq=2034, feature_extraction=False, downsample_factor=1, normalization_technique="minmax"):
    """
    Main Pre-processing Pipeline.
    
    Parameters:
    - scan: numpy array of shape (#electrodes, #timesteps)
    - sfreq: Sampling frequency of the raw acquisition machine (Default: 2034Hz)
    """
    ch_names = [f'MEG_{i:03d}' for i in range(scan.shape[0])]
    ch_types = ['mag'] * scan.shape[0]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    raw = mne.io.RawArray(scan, info)
    denoised = noise_reduction(raw) # Apply noise reduction before downsampling to preserve signal quality for feature extraction, especially for Fourier and Wavelet analysis which are sensitive to noise.
    downsampled = downsample(denoised, downsample_factor) # Downsample after noise reduction to preserve as much signal quality as possible for the Fourier Transform and Wavelet analysis, which are sensitive to noise. This way we reduce the data size for the model while keeping the most informative features intact.
    zero_meaned = zero_mean(downsampled) # Shape: (electrodes, timesteps)

    # If only normalization is requested, skip noise reduction to save time
    if feature_extraction is False:
        return normalization(zero_meaned.get_data(), normalization_technique)
        
    elif feature_extraction.lower() == "fourier":
        spectrum = zero_meaned.compute_psd(method='welch', fmin=0.5, fmax=100.0)
        psd, frequencies = spectrum.get_data(return_freqs=True)

        aggregated_bands = aggregate_bands(psd, frequencies, data_type="fourier")
        print(f"Aggregated band powers shape: {aggregated_bands.shape} (electrodes, bands)")
        
        return aggregated_bands
        
    elif feature_extraction.lower() == "wavelets":
        # Use denser sampling at low frequencies and wider steps at high frequencies
        # Align sampling with the band definitions used in `aggregate_bands`
        delta = np.arange(1, 4, 1)          # 1-3 Hz
        theta = np.arange(4, 8, 1)          # 4-7 Hz
        alpha = np.arange(8, 12, 2)         # 8-11 Hz (step 2 Hz)
        beta = np.arange(12, 30, 4)         # 12-28 Hz (step 4 Hz)
        gamma = np.arange(30, 101, 10)       # 30-100 Hz (step 10 Hz)
        frequencies = np.concatenate([delta, theta, alpha, beta, gamma])
        epochs = mne.EpochsArray(zero_meaned.get_data()[np.newaxis, ...], zero_meaned.info, verbose=False)
        
        # Use more cycles at low frequencies for better frequency resolution,
        # and fewer cycles at high frequencies for better temporal resolution.
        # Use logarithmic spacing so cycles decrease faster across frequency.
        n_cycles = np.logspace(np.log10(8.0), np.log10(3.0), len(frequencies))
        tfr = mne.time_frequency.tfr_morlet(
            epochs, freqs=frequencies, n_cycles=n_cycles,
            return_itc=False, average=True, verbose=False
        )
        
        if tfr.data.ndim == 4:
            tfr_data = np.squeeze(tfr.data, axis=0) 
        else:
            tfr_data = tfr.data

        aggregated_bands = aggregate_bands(tfr_data, frequencies, data_type="wavelets")
        #print(f"Aggregated band powers shape: {aggregated_bands.shape} (electrodes, bands, time)")
                
        return normalization_3d(aggregated_bands, normalization_technique)

    else:
        raise ValueError("Invalid feature_extraction option. Choose False, 'Fourier', or 'Wavelets'.")