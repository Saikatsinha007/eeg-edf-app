import streamlit as st
import mne
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pywt
import tempfile
import pandas as pd
from transformers import pipeline
import plotly.graph_objects as go
from mne.preprocessing import ICA
from mne_connectivity import spectral_connectivity
from mne.viz import plot_topomap
from scipy.stats import skew, kurtosis

# Load an open-source LLM for medical text generation
llm = pipeline("text-generation", model="gpt-4")  # Replace with GPT-4 if available

# Load a pre-trained EEG classifier (Replace with your own model)
class EEGClassifier(nn.Module):
    def __init__(self):
        super(EEGClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 3)  # Example output classes
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # Change shape for LSTM
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Take last output of LSTM
        return x

model = EEGClassifier()

# Streamlit UI
st.set_page_config(page_title="EEG Analyzer", page_icon="🧠", layout="wide")
st.title("🧠 AI-Powered EEG Analyzer with Advanced Features for Doctors")

st.sidebar.header("📂 Upload streamlitEEG Data")
uploaded_file = st.sidebar.file_uploader("Upload EDF file", type=["edf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmpfile:
        tmpfile.write(uploaded_file.read())
        tmpfile_path = tmpfile.name
    
    raw = mne.io.read_raw_edf(tmpfile_path, preload=True)
    st.write("### 🏷️ EEG Channels:", raw.ch_names)
    
    # EEG Signal Visualization
    st.write("### 📊 EEG Signal Visualization")
    fig = raw.plot(duration=5, scalings='auto', show=False)
    st.pyplot(fig)
    
    # Apply band-pass filtering to remove noise and extract relevant signals
    sfreq = raw.info['sfreq']
    raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=min(49.9, sfreq / 2 - 0.1), fir_design='firwin')
    
    # Artifact Detection and Removal using ICA
    st.write("### 🧹 Artifact Detection and Removal (ICA)")
    ica = ICA(n_components=15, random_state=97)
    ica.fit(raw_filtered)
    ica.plot_sources(raw_filtered, show=False)
    st.pyplot(plt.gcf())
    ica.plot_components(show=False)
    st.pyplot(plt.gcf())
    
    # Compute Power Spectral Density (PSD)
    st.write("### 🔬 Power Spectral Density (PSD)")
    psd = raw_filtered.compute_psd(method='welch', fmin=0.5, fmax=50, n_fft=2048, verbose=False)
    psds = psd.get_data()
    freqs = psd.freqs
    mean_psd = np.mean(psds, axis=0)
    
    # Extract frequency bands
    bands = {
        "Delta (0.5-4 Hz)": (0.5, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-12 Hz)": (8, 12),
        "Beta (12-30 Hz)": (12, 30),
        "Gamma (30-50 Hz)": (30, 50),
    }
    
    band_psd = {
        band: np.mean(mean_psd[np.logical_and(freqs >= low, freqs < high)]) 
        for band, (low, high) in bands.items()
    }
    
    df_band = pd.DataFrame(list(band_psd.items()), columns=['Band', 'Power'])
    st.table(df_band)

    # **Plot PSD for all frequencies**
    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(x=freqs, y=mean_psd, mode='lines', name='PSD'))
    fig_psd.update_layout(title="PSD across EEG Frequencies", xaxis_title="Frequency (Hz)", yaxis_title="Power Spectral Density")
    st.plotly_chart(fig_psd)

    # Time-Frequency Analysis using Continuous Wavelet Transform (CWT)
    st.write("### ⏳ Time-Frequency Analysis (CWT)")
    data, times = raw_filtered[:1, :1000]
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(data[0], scales, 'cmor', sampling_period=1/sfreq)
    fig_cwt = go.Figure(data=go.Heatmap(z=np.abs(coefficients), x=times, y=frequencies, colorscale='Viridis'))
    fig_cwt.update_layout(title="Continuous Wavelet Transform (CWT)", xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
    st.plotly_chart(fig_cwt)

    # Connectivity Analysis
    st.write("### 🔗 Functional Connectivity Analysis")
    con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        raw_filtered, method='coh', mode='multitaper', sfreq=sfreq, fmin=0.5, fmax=50
    )
    fig_con = go.Figure(data=go.Heatmap(z=con[:, :, 0], x=freqs, y=raw.ch_names, colorscale='Viridis'))
    fig_con.update_layout(title="Functional Connectivity (Coherence)", xaxis_title="Frequency (Hz)", yaxis_title="Channels")
    st.plotly_chart(fig_con)

    # Spatial Mapping (Topomap)
    st.write("### 🗺️ Spatial Mapping (Topomap)")
    fig_topomap, ax = plt.subplots(figsize=(8, 4))
    plot_topomap(mean_psd, raw.info, axes=ax, show=False)
    st.pyplot(fig_topomap)

    # Statistical Analysis
    st.write("### 📊 Statistical Analysis")
    stats = {
        "Mean": np.mean(raw_filtered.get_data()),
        "Variance": np.var(raw_filtered.get_data()),
        "Skewness": skew(raw_filtered.get_data().flatten()),
        "Kurtosis": kurtosis(raw_filtered.get_data().flatten()),
    }
    df_stats = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
    st.table(df_stats)

    # Feature Extraction using Wavelet Transform
    st.write("### ⚡ Feature Extraction with Wavelet Transform")
    data, times = raw_filtered[:1, :1000]  # Use the first channel and first 1000 samples
    coeffs = pywt.wavedec(data[0], 'db4', level=4)  # Decompose the signal using Wavelet Transform
    feature_vector = np.hstack([np.mean(abs(c)) for c in coeffs])  # Compute mean absolute values of coefficients

    # Pad or truncate the feature vector to a fixed length (e.g., 100)
    feature_vector = np.pad(feature_vector, (0, 100 - len(feature_vector)), 'constant') if len(feature_vector) < 100 else feature_vector[:100]

    # Display the feature vector
    st.write("#### Extracted Feature Vector:")
    st.write(feature_vector)

    # AI Model Classification
    st.write("### 🤖 AI Classification")
    data_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Convert to tensor and add batch and channel dimensions
    prediction = model(data_tensor)
    predicted_class = torch.argmax(prediction, dim=1).item()
    st.success(f"### 🎯 AI Classification Result: Class {predicted_class}")
    
    # LLM-Generated Report
    st.write("### 📜 AI-Generated EEG Report")
    report_prompt = f"""
    Generate a detailed medical report based on EEG analysis.
    Channels Analyzed: {', '.join(raw.ch_names)}
    AI Classification: Class {predicted_class}
    Power Spectral Analysis: {band_psd}
    Functional Connectivity: {con[:, :, 0]}
    Statistical Analysis: {stats}
    Provide an expert-level report explaining possible clinical implications.
    """
    
    generated_report = llm(report_prompt, max_length=500)[0]['generated_text']
    st.text_area("Generated Report:", generated_report, height=200)
    
    if st.button("📄 Download Report"):
        st.download_button(label="Download Report", data=generated_report, file_name="EEG_Report.txt", mime="text/plain")
    
    # LLM Chat for Doctor Queries
    st.write("### 🏥 AI Medical Assistant - Ask Questions about the EEG Report")
    user_query = st.text_input("Ask about the EEG results:")
    if user_query:
        response = llm(user_query, max_length=200)[0]['generated_text']
        st.write("🤖 AI Response:", response)