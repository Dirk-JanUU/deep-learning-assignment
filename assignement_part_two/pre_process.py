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

def normalization_3d(wavelet_output, technique="zscore"):
    """
    Normalizes a 3D wavelet array per frequency band.
    
    Parameters:
    - wavelet_output: numpy array of shape (electrodes, bands, time)
    - technique: "minmax" or "zscore"
    
    Returns:
    - normalized array of shape (electrodes, bands, time)
    """
    output = wavelet_output.copy()
    
    for b in range(output.shape[1]):
        band_slice = output[:, b, :]  # (electrodes, time)
        output[:, b, :] = normalization(band_slice, technique)
    
    return output

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

    # If only normalization is requested, skip noise reduction to save time
    if feature_extraction is False:
        if downsample_factor > 1:
            raw = downsample(raw, downsample_factor)

        processed_matrix = raw.get_data()
        output = normalization(processed_matrix, normalization_technique)
        return output
        
    elif feature_extraction.lower() == "fourier":
        raw = noise_reduction(raw)
        if downsample_factor > 1:
            raw = downsample(raw, downsample_factor)
        spectrum = raw.compute_psd(method='welch', fmin=0.5, fmax=100.0)
        psd, frequencies = spectrum.get_data(return_freqs=True)

        aggregated_bands = aggregate_bands(psd, frequencies, data_type="fourier")
        print(f"Aggregated band powers shape: {aggregated_bands.shape} (electrodes, bands)")
        
        return aggregated_bands
        
    elif feature_extraction.lower() == "wavelets":
        raw = noise_reduction(raw)
        if downsample_factor > 1:
            raw = downsample(raw, downsample_factor)
        processed_matrix = raw.get_data()
        # frequencies = np.arange(1, 101, 1)
        frequencies = np.array([
            # Delta
            1, 2, 3,
            # Theta
            5, 6, 7,
            # Alpha
            9, 10, 11,
            # Beta
            15, 20, 25,
            # Gamma
            50, 70, 90
        ])
        epochs = mne.EpochsArray(processed_matrix[np.newaxis, ...], raw.info, verbose=False)
        
        tfr = mne.time_frequency.tfr_morlet(
            epochs, freqs=frequencies, n_cycles=frequencies / 2, 
            return_itc=False, average=True, verbose=False
        )
        
        if tfr.data.ndim == 4:
            tfr_data = np.squeeze(tfr.data, axis=0) 
        else:
            tfr_data = tfr.data

        aggregated_bands = aggregate_bands(tfr_data, frequencies, data_type="wavelets")
        print(f"Aggregated band powers shape: {aggregated_bands.shape} (electrodes, bands, time)")
                
        return normalization_3d(aggregated_bands, normalization_technique)

    else:
        raise ValueError("Invalid feature_extraction option. Choose False, 'Fourier', or 'Wavelets'.")