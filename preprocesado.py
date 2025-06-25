import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def bandpass_filter(signal, low=0.5, high=40, fs=250):
    """Aplicar filtro pasa banda a la señal ECG"""
    b, a = butter(2, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, signal)

def extract_features(ecg_signal):
    """Extraer características de la señal ECG con detección mejorada"""
    signal_range = np.max(ecg_signal) - np.min(ecg_signal)
    min_height = 0.4 * signal_range if signal_range > 0 else 0.5
    
    peaks, properties = find_peaks(ecg_signal, height=min_height, distance=50, prominence=0.3)
    
    if len(peaks) < 2:
        return {
            "num_peaks": len(peaks),
            "mean_rr_interval": 0,
            "std_rr_interval": 0,
            "mean_amplitude": np.mean(ecg_signal) if len(ecg_signal) > 0 else 0,
            "std_amplitude": np.std(ecg_signal) if len(ecg_signal) > 0 else 0,
            "max_amplitude": np.max(ecg_signal) if len(ecg_signal) > 0 else 0,
            "min_amplitude": np.min(ecg_signal) if len(ecg_signal) > 0 else 0,
            "signal_energy": np.sum(np.square(ecg_signal)) if len(ecg_signal) > 0 else 0,
            "peak_prominences": 0
        }
    
    rr_intervals = np.diff(peaks)
    prominences = properties.get('prominences', np.zeros_like(peaks))
    
    features = {
        "num_peaks": len(peaks),
        "mean_rr_interval": float(np.mean(rr_intervals)),
        "std_rr_interval": float(np.std(rr_intervals)),
        "rr_variability": float(np.std(rr_intervals) / np.mean(rr_intervals)) if np.mean(rr_intervals) > 0 else 0,
        "mean_amplitude": float(np.mean(ecg_signal)),
        "std_amplitude": float(np.std(ecg_signal)),
        "max_amplitude": float(np.max(ecg_signal)),
        "min_amplitude": float(np.min(ecg_signal)),
        "signal_energy": float(np.sum(np.square(ecg_signal))),
        "peak_prominences": float(np.mean(prominences)) if len(prominences) > 0 else 0
    }
    return features

def segment_signal(signal, segment_length=500, overlap=0.5):
    """Segmenta la señal en fragmentos con superposición"""
    segments = []
    step = int(segment_length * (1 - overlap))
    for i in range(0, len(signal) - segment_length + 1, step):
        segment = signal[i:i + segment_length]
        segments.append(segment)
    return np.array(segments)