import streamlit as st
import mne
import numpy as np
import matplotlib.pyplot as plt
# Placeholder for the pre-defined seizure prediction model
# Replace this with the actual model import/function
def predict_seizure(data):
    # Dummy implementation: assumes seizure if mean amplitude exceeds a threshold
    # Replace with your actual model
    return np.mean(np.abs(data)) > 0.0001  # Example threshold (in volts)

# Streamlit app title and description
st.title("EEG EDF File Analyzer")
st.markdown("""
Upload an EEG EDF file to analyze it for seizure prediction and power spectral density (PSD).
This tool provides:
- Seizure prediction using a pre-defined model.
- PSD analysis for alpha (8-12 Hz), beta (12-30 Hz), and gamma (30-100 Hz) bands.
""")

# File uploader
uploaded_file = st.file_uploader("Upload an EDF file", type=["edf"])

if uploaded_file is not None:
    try:
        # Read the EDF file using MNE
        raw = mne.io.read_raw_edf(uploaded_file, preload=True)
        st.success("File loaded successfully!")

        # Extract EEG data as a NumPy array (channels x samples)
        data = raw.get_data()

        # --- Seizure Prediction ---
        prediction = predict_seizure(data)
        st.subheader("Seizure Prediction")
        st.write(f"Seizure predicted: **{'Yes' if prediction else 'No'}**")

        # --- PSD Calculation ---
        # Compute PSD using Welch's method (fmax=100 Hz to cover gamma band)
        psd, freqs = raw.compute_psd(fmin=0, fmax=100, n_fft=2048, dB=False)
        psd_mean = psd.mean(axis=0)  # Average PSD across all channels
        df = freqs[1] - freqs[0]    # Frequency resolution

        # Define frequency bands
        bands = {
            "alpha": (8, 12),
            "beta": (12, 30),
            "gamma": (30, 100)
        }

        # Calculate power in each band
        band_powers = {}
        for band, (fmin, fmax) in bands.items():
            idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
            band_powers[band] = np.sum(psd_mean[idx]) * df

        # --- PSD Visualization ---
        st.subheader("Power Spectral Density (PSD)")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(freqs, psd_mean, color='blue')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (V²/Hz)")
        ax.set_title("Average PSD Across All Channels")

        # Shade frequency bands
        for band, (fmin, fmax) in bands.items():
            ax.axvspan(fmin, fmax, alpha=0.2, label=band.capitalize())
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # --- Display Band Powers ---
        st.subheader("Band Powers")
        st.markdown("Power in each frequency band (averaged across channels):")
        for band, power in band_powers.items():
            st.write(f"**{band.capitalize()} ({bands[band][0]}-{bands[band][1]} Hz):** {power:.2e} V²")

    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
else:
    st.info("Please upload an EDF file to begin analysis.")

# Footer
st.markdown("---")
st.write("Developed for EEG analysis and seizure prediction.")