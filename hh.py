import streamlit as st
import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, lfilter, welch
from PIL import Image
import tempfile
import io
import zipfile
import torch
import torch.nn as nn
import math
from scipy import stats

# Set page configuration
st.set_page_config(page_title="Advanced EEG & Image Analysis", layout="wide")

# Signal processing functions
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def calculate_psd(signal, fs):
    freqs, psd = welch(signal, fs, nperseg=1024)
    return freqs, psd

def calculate_band_psd(signal, fs):
    """Calculate PSD for standard frequency bands"""
    freqs, psd = welch(signal, fs, nperseg=1024)
    
    # Define frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 100)
    }
    
    band_powers = {}
    for band, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_powers[band] = np.trapz(psd[idx], freqs[idx])
    
    return freqs, psd, band_powers

# Artifact detection functions
def detect_artifacts_statistical(signal, window_size=256, threshold=3.0):
    """Detect artifacts using statistical methods"""
    n_windows = len(signal) // window_size
    artifact_windows = []
    
    for i in range(n_windows):
        segment = signal[i*window_size:(i+1)*window_size]
        z_scores = np.abs(stats.zscore(segment))
        if np.any(z_scores > threshold):
            artifact_windows.append((i*window_size, (i+1)*window_size))
    
    return artifact_windows

# CNN Model for artifact detection
class ArtifactDetector(nn.Module):
    def __init__(self, input_size=256):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=32, stride=4, padding=8)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=16, stride=2, padding=4)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self._calculate_fc_size(input_size)
        
    def _calculate_fc_size(self, input_size):
        size = math.floor((input_size + 2*8 - 32) / 4 + 1)
        size = math.floor((size + 2*4 - 16) / 2 + 1)
        size = 16  # Adaptive pooling
        self.fc_size = 32 * size
        self.fc = nn.Linear(self.fc_size, 2)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def predict_artifacts_cnn(signal, model, window_size=256):
    """Predict artifacts using sliding window CNN approach"""
    n_windows = len(signal) // window_size
    artifact_regions = []
    
    for i in range(n_windows):
        segment = signal[i*window_size:(i+1)*window_size]
        if len(segment) < window_size:
            continue
            
        segment_tensor = torch.tensor(segment, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            output = model(segment_tensor)
            prob = torch.softmax(output, dim=1)[0,1].item()
            
        if prob > 0.7:  # Threshold for artifact detection
            artifact_regions.append((i*window_size, (i+1)*window_size, prob))
    
    return artifact_regions

# Visualization functions
def plot_signal_with_artifacts(time, signal, artifact_regions, title="EEG Signal with Artifacts"):
    fig = go.Figure()
    
    # Plot the clean signal
    fig.add_trace(go.Scatter(
        x=time,
        y=signal,
        mode='lines',
        name='EEG Signal',
        line=dict(color='blue')
    ))
    
    # Highlight artifact regions
    for start, end, prob in artifact_regions:
        fig.add_vrect(
            x0=time[start],
            x1=time[end],
            fillcolor="red",
            opacity=0.2,
            line_width=0,
            annotation_text=f"Artifact ({prob:.2f})",
            annotation_position="top left"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Amplitude (µV)',
        hovermode="x unified"
    )
    return fig

# Main application
pages = ["EEG Signal Analysis", "Image Analysis"]
page_selection = st.sidebar.selectbox("Choose Analysis Type", pages)

if page_selection == "EEG Signal Analysis":
    st.title('Advanced EEG Signal Analysis with Artifact Localization')
    uploaded_file = st.sidebar.file_uploader("Choose EEG File", type=["edf"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.edf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmpfile:
                tmpfile.write(uploaded_file.getbuffer())
                tmpfile_path = tmpfile.name
            
            raw = mne.io.read_raw_edf(tmpfile_path, preload=True)
            duration = 10
            raw.crop(tmin=0, tmax=duration)
            data, times = raw[:, :]

            df = pd.DataFrame(data.T, columns=raw.ch_names)
            fs = int(raw.info['sfreq'])
        else:
            st.error("Unsupported file format. Please upload an EDF file.")
            st.stop()

        selected_channel = st.sidebar.selectbox('Select EEG Channel', df.columns.tolist())
        signal = df[selected_channel].values
        time = np.arange(len(signal)) / fs

        # Bandpass Filtering
        st.subheader('Signal Processing')
        col1, col2 = st.columns(2)
        lowcut = col1.slider('Lowcut Frequency (Hz)', 0.1, 20.0, 1.0)
        highcut = col2.slider('Highcut Frequency (Hz)', 1.0, 100.0, 30.0)
        filtered_signal = bandpass_filter(signal, lowcut, highcut, fs)

        # Artifact Detection
        st.subheader('Artifact Detection')
        detection_method = st.radio("Detection Method", 
                                   ["Statistical (Z-score)", "CNN Model"], 
                                   horizontal=True)
        
        artifact_regions = []
        
        if detection_method == "Statistical (Z-score)":
            threshold = st.slider('Z-score Threshold', 1.0, 5.0, 3.0, 0.1)
            window_size = st.slider('Window Size (samples)', 64, 1024, 256)
            artifact_windows = detect_artifacts_statistical(filtered_signal, window_size, threshold)
            artifact_regions = [(start, end, 1.0) for start, end in artifact_windows]
        else:
            # Initialize CNN model
            model = ArtifactDetector(input_size=256)
            window_size = st.slider('Window Size (samples)', 64, 1024, 256)
            artifact_regions = predict_artifacts_cnn(filtered_signal, model, window_size)

        # Visualization
        st.subheader('Artifact Visualization')
        fig = plot_signal_with_artifacts(time, filtered_signal, artifact_regions)
        st.plotly_chart(fig, use_container_width=True)

        # Artifact Summary
        if artifact_regions:
            st.subheader('Artifact Summary')
            artifact_df = pd.DataFrame([
                {
                    "Start Time (s)": time[start],
                    "End Time (s)": time[end],
                    "Duration (s)": time[end] - time[start],
                    "Confidence": prob
                }
                for start, end, prob in artifact_regions
            ])
            st.dataframe(artifact_df.style.format({
                "Start Time (s)": "{:.3f}",
                "End Time (s)": "{:.3f}",
                "Duration (s)": "{:.3f}",
                "Confidence": "{:.2f}"
            }))

            # Option to remove artifacts
            if st.checkbox('Show artifact-removed signal'):
                clean_signal = filtered_signal.copy()
                for start, end, _ in artifact_regions:
                    clean_signal[start:end] = np.nan
                
                fig_clean = go.Figure()
                fig_clean.add_trace(go.Scatter(
                    x=time,
                    y=clean_signal,
                    mode='lines',
                    name='Artifact-Removed Signal',
                    line=dict(color='green')
                ))
                fig_clean.update_layout(
                    title='EEG Signal with Artifacts Removed',
                    xaxis_title='Time (s)',
                    yaxis_title='Amplitude (µV)'
                )
                st.plotly_chart(fig_clean, use_container_width=True)
        else:
            st.success("No artifacts detected in the signal!")

        # PSD Analysis
        st.subheader('Power Spectral Density Analysis')
        
        # Calculate band powers
        freqs, psd, band_powers = calculate_band_psd(filtered_signal, fs)
        
        # Show band powers in a table
        st.write("### Frequency Band Powers")
        band_df = pd.DataFrame.from_dict(band_powers, orient='index', columns=['Power'])
        st.dataframe(band_df.style.format("{:.2f}"))
        
        # Plot full PSD
        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', name='PSD'))
        
        # Add shaded regions for frequency bands
        bands = {
            'Delta': (0.5, 4, 'rgba(0, 0, 255, 0.1)'),
            'Theta': (4, 8, 'rgba(0, 255, 0, 0.1)'),
            'Alpha': (8, 13, 'rgba(255, 0, 0, 0.1)'),
            'Beta': (13, 30, 'rgba(255, 255, 0, 0.1)'),
            'Gamma': (30, 100, 'rgba(255, 0, 255, 0.1)')
        }
        
        for band, (low, high, color) in bands.items():
            fig_psd.add_vrect(
                x0=low, x1=high,
                fillcolor=color,
                opacity=0.2,
                line_width=0,
                annotation_text=band,
                annotation_position="top left"
            )
        
        fig_psd.update_layout(
            title=f'{selected_channel} Power Spectrum with Frequency Bands',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Power Spectral Density (dB/Hz)',
            xaxis_range=[0, 50]  # Focus on relevant frequencies
        )
        st.plotly_chart(fig_psd, use_container_width=True)
        
        # Add band-specific PSD plots
        st.write("### Band-Specific Power Spectra")
        band_choice = st.selectbox("Select frequency band to view", list(bands.keys()))
        
        low, high, _ = bands[band_choice]
        idx = np.logical_and(freqs >= low, freqs <= high)
        
        fig_band = go.Figure()
        fig_band.add_trace(go.Scatter(
            x=freqs[idx],
            y=psd[idx],
            mode='lines',
            name=f'{band_choice} Power'
        ))
        fig_band.update_layout(
            title=f'{band_choice} Band ({low}-{high}Hz) Power Spectrum',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Power Spectral Density (dB/Hz)'
        )
        st.plotly_chart(fig_band, use_container_width=True)

elif page_selection == "Image Analysis":
    st.title('Advanced Image Analysis')
    uploaded_image = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

st.sidebar.markdown("---")
st.sidebar.info("ℹ️ Select EEG analysis for neural signal processing or Image analysis for computer vision tasks.")