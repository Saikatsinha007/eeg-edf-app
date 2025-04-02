# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.signal import butter, lfilter, welch
# from scipy.ndimage import gaussian_filter1d
# from matplotlib.backends.backend_pdf import PdfPages
# import io
# import plotly.graph_objects as go


# # Function to apply a Butterworth bandpass filter
# def butter_bandpass(lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return b, a

# # Function to filter data using the Butterworth bandpass filter
# def bandpass_filter(data, lowcut, highcut, fs, order=5):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, data)
#     return y

# # Function to calculate Power Spectral Density (PSD)
# def calculate_psd(signal, fs):
#     freqs, psd = welch(signal, fs, nperseg=1024)
#     return freqs, psd

# # Title of the app
# st.title('Advanced EEG Signal Analysis and Visualization')

# # Sidebar for user inputs
# st.sidebar.title("Settings")

# # File uploader to allow user to upload their EEG data
# uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# # Default settings
# default_fs = 256
# default_bands = {
#     'Delta': (0.5, 4),
#     'Theta': (4, 8),
#     'Alpha': (8, 13),
#     'Beta': (13, 30),
#     'Gamma': (30, 45)
# }

# # Sampling frequency input
# fs = st.sidebar.number_input('Sampling Frequency (Hz)', value=default_fs)

# # Frequency bands for EEG waves
# st.sidebar.subheader('Frequency Bands')
# bands = {}
# for band, (low, high) in default_bands.items():
#     bands[band] = (
#         st.sidebar.number_input(f'{band} Wave Lower Bound (Hz)', value=low),
#         st.sidebar.number_input(f'{band} Wave Upper Bound (Hz)', value=high)
#     )

# # Signal Smoothing option
# st.sidebar.subheader('Signal Smoothing')
# smoothing = st.sidebar.checkbox('Apply Smoothing')
# smoothing_sigma = st.sidebar.slider('Smoothing Sigma', min_value=0.1, max_value=10.0, value=2.0)

# if uploaded_file is not None:
#     # Read the uploaded CSV file
#     df = pd.read_csv(uploaded_file)
    
#     # Show a preview of the data
#     st.subheader('EEG Signal Data Preview')
#     st.write(df.head())

#     # EEG Signal Visualization
#     st.subheader('EEG Signal Visualization')
#     columns = df.columns.tolist()
#     selected_channel = st.sidebar.selectbox('Select EEG Channel', columns)

#     if selected_channel:
#         signal = df[selected_channel].values
#         time = np.arange(len(signal)) / fs
        
#         if smoothing:
#             signal = gaussian_filter1d(signal, sigma=smoothing_sigma)

#         # Initialize a PDF file
#         pdf_buffer = io.BytesIO()
#         pdf_pages = PdfPages(pdf_buffer)

#         fig, ax = plt.subplots()
#         ax.plot(time, signal)
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('Amplitude')
#         ax.set_title(f'{selected_channel} Signal')
#         st.pyplot(fig)
#         pdf_pages.savefig(fig)
        
#         # Power Spectral Density (PSD) Analysis
#         st.subheader('Power Spectral Density (PSD) Analysis')
#         if selected_channel:
#             freqs, psd = calculate_psd(signal, fs)
#             fig, ax = plt.subplots()
#             ax.semilogy(freqs, psd)
#             ax.set_xlabel('Frequency (Hz)')
#             ax.set_ylabel('Power/Frequency (dB/Hz)')
#             ax.set_title(f'{selected_channel} Power Spectral Density')
#             st.pyplot(fig)
#             pdf_pages.savefig(fig)

#         # Visualize different EEG waves
#         for band, (low, high) in bands.items():
#             st.subheader(f'{band} Wave Visualization ({low}-{high} Hz)')
#             if selected_channel:
#                 filtered_signal = bandpass_filter(signal, low, high, fs)
#                 if smoothing:
#                     filtered_signal = gaussian_filter1d(filtered_signal, sigma=smoothing_sigma)
#                 fig, ax = plt.subplots()
#                 ax.plot(time, filtered_signal)
#                 ax.set_xlabel('Time (s)')
#                 ax.set_ylabel('Amplitude')
#                 ax.set_title(f'{selected_channel} {band} Waves ({low}-{high} Hz)')
#                 st.pyplot(fig)
#                 pdf_pages.savefig(fig)
        
#         # Overlay Multiple EEG Waves
#         st.subheader('Overlay Multiple EEG Waves')
#         overlay_waves = st.multiselect('Select Waves to Overlay', bands.keys(), default=list(bands.keys()))
#         if overlay_waves:
#             fig, ax = plt.subplots()
#             for wave in overlay_waves:
#                 low, high = bands[wave]
#                 filtered_signal = bandpass_filter(signal, low, high, fs)
#                 if smoothing:
#                     filtered_signal = gaussian_filter1d(filtered_signal, sigma=smoothing_sigma)
#                 ax.plot(time, filtered_signal, label=f'{wave} ({low}-{high} Hz)')
#             ax.set_xlabel('Time (s)')
#             ax.set_ylabel('Amplitude')
#             ax.set_title(f'{selected_channel} Overlayed EEG Waves')
#             ax.legend()
#             st.pyplot(fig)
#             pdf_pages.savefig(fig)

#         # Custom Time Range Selection
#         st.subheader('Custom Time Range Selection')
#         time_min, time_max = st.slider('Select Time Range (seconds)', min_value=0.0, max_value=float(time[-1]), value=(0.0, float(time[-1])))
#         time_range_signal = signal[int(time_min*fs):int(time_max*fs)]
#         time_range_time = time[int(time_min*fs):int(time_max*fs)]

#         fig, ax = plt.subplots()
#         ax.plot(time_range_time, time_range_signal)
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('Amplitude')
#         ax.set_title(f'{selected_channel} Signal from {time_min} to {time_max} seconds')
#         st.pyplot(fig)
#         pdf_pages.savefig(fig)

#         # Summary statistics for selected channel
#         st.subheader(f'Summary Statistics for {selected_channel}')
#         st.write(df[selected_channel].describe())
        
#         # Option to download filtered signals
#         st.sidebar.subheader('Download Filtered Signals')
#         download_options = {}
#         for band in bands.keys():
#             download_options[band] = st.sidebar.checkbox(f'Download {band} Waves')
        
#         for band, (low, high) in bands.items():
#             if download_options[band]:
#                 filtered_signal = bandpass_filter(signal, low, high, fs)
#                 if smoothing:
#                     filtered_signal = gaussian_filter1d(filtered_signal, sigma=smoothing_sigma)
#                 signal_csv = pd.DataFrame({f'{selected_channel}_{band.lower()}': filtered_signal})
#                 st.sidebar.download_button(f'Download {band} CSV', signal_csv.to_csv(index=False), f'{selected_channel}_{band.lower()}.csv')

#         # Save the PDF and create a download button
#         pdf_pages.close()
#         pdf_buffer.seek(0)
#         st.sidebar.download_button('Download All Plots as PDF', pdf_buffer, 'EEG_signal_plots.pdf')

# # Instructions for the user
# st.write("Upload an EEG data CSV file to start visualizing the signals and use the sidebar to adjust settings.")



import streamlit as st
import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, lfilter, welch
from scipy.ndimage import gaussian_filter1d
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Advanced EEG & Image Analysis", layout="wide")

# Function to apply a Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to filter data using the Butterworth bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Function to calculate Power Spectral Density (PSD)
def calculate_psd(signal, fs):
    freqs, psd = welch(signal, fs, nperseg=1024)
    return freqs, psd

# Function to detect artifacts (e.g., eye blinks, muscle activity)
def detect_artifacts(signal, fs):
    threshold = 3 * np.std(signal)
    artifacts = np.abs(signal) > threshold
    return artifacts

# Define the pages of the app
pages = ["EEG Signal Analysis", "Image Analysis"]

# Sidebar for page selection
page_selection = st.sidebar.selectbox("Choose Analysis Type", pages)

# EEG Signal Analysis Page
if page_selection == "EEG Signal Analysis":
    st.title('Advanced EEG Signal Analysis and Visualization')
    
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "edf"])

    default_fs = 256
    default_bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }

    fs = st.sidebar.number_input('Sampling Frequency (Hz)', value=default_fs)

    st.sidebar.subheader('Frequency Bands')
    bands = {}
    for band, (low, high) in default_bands.items():
        bands[band] = (
            st.sidebar.number_input(f'{band} Wave Lower Bound (Hz)', value=low),
            st.sidebar.number_input(f'{band} Wave Upper Bound (Hz)', value=high)
        )

    st.sidebar.subheader('Signal Smoothing')
    smoothing = st.sidebar.checkbox('Apply Smoothing')
    smoothing_sigma = st.sidebar.slider('Smoothing Sigma', min_value=0.1, max_value=10.0, value=2.0)

    st.sidebar.subheader('Artifacts Detection')
    detect_artifacts_button = st.sidebar.button('Detect Artifacts')

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.edf'):
            raw = mne.io.read_raw_edf(uploaded_file, preload=True)
            data = raw.get_data()
            df = pd.DataFrame(data.T, columns=raw.ch_names)

        st.subheader('EEG Signal Data Preview')
        st.write(df.head())

        st.subheader('EEG Signal Visualization')
        columns = df.columns.tolist()
        selected_channel = st.sidebar.selectbox('Select EEG Channel', columns)

        if selected_channel:
            signal = df[selected_channel].values
            time = np.arange(len(signal)) / fs
            
            if smoothing:
                signal = gaussian_filter1d(signal, sigma=smoothing_sigma)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name='Signal'))
            fig.update_layout(title=f'{selected_channel} Signal',
                              xaxis_title='Time (s)',
                              yaxis_title='Amplitude')
            st.plotly_chart(fig)
            
            st.subheader('Power Spectral Density (PSD) Analysis')
            if selected_channel:
                freqs, psd = calculate_psd(signal, fs)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', name='PSD'))
                fig.update_layout(title=f'{selected_channel} Power Spectral Density',
                                  xaxis_title='Frequency (Hz)',
                                  yaxis_title='Power/Frequency (dB/Hz)')
                st.plotly_chart(fig)

            if detect_artifacts_button:
                st.subheader('Artifacts Detected')
                artifacts = detect_artifacts(signal, fs)
                artifact_times = time[artifacts]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name='Signal'))
                fig.add_trace(go.Scatter(x=artifact_times, y=signal[artifacts], mode='markers', name='Artifacts', marker=dict(color='red')))
                fig.update_layout(title=f'{selected_channel} Signal with Artifacts',
                                  xaxis_title='Time (s)',
                                  yaxis_title='Amplitude')
                st.plotly_chart(fig)

            for band, (low, high) in bands.items():
                st.subheader(f'{band} Wave Visualization ({low}-{high} Hz)')
                if selected_channel:
                    filtered_signal = bandpass_filter(signal, low, high, fs)
                    if smoothing:
                        filtered_signal = gaussian_filter1d(filtered_signal, sigma=smoothing_sigma)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=time, y=filtered_signal, mode='lines', name=f'{band} Waves'))
                    fig.update_layout(title=f'{selected_channel} {band} Waves ({low}-{high} Hz)',
                                      xaxis_title='Time (s)',
                                      yaxis_title='Amplitude')
                    st.plotly_chart(fig)
            
            st.subheader('Overlay Multiple EEG Waves')
            overlay_waves = st.multiselect('Select Waves to Overlay', bands.keys(), default=list(bands.keys()))
            if overlay_waves:
                fig = go.Figure()
                for wave in overlay_waves:
                    low, high = bands[wave]
                    filtered_signal = bandpass_filter(signal, low, high, fs)
                    if smoothing:
                        filtered_signal = gaussian_filter1d(filtered_signal, sigma=smoothing_sigma)
                    fig.add_trace(go.Scatter(x=time, y=filtered_signal, mode='lines', name=f'{wave} ({low}-{high} Hz)'))
                fig.update_layout(title=f'{selected_channel} Overlayed EEG Waves',
                                  xaxis_title='Time (s)',
                                  yaxis_title='Amplitude')
                st.plotly_chart(fig)

            st.subheader('Custom Time Range Selection')
            time_min, time_max = st.slider('Select Time Range (seconds)', min_value=0.0, max_value=float(time[-1]), value=(0.0, float(time[-1])))
            time_range_signal = signal[int(time_min*fs):int(time_max*fs)]
            time_range_time = time[int(time_min*fs):int(time_max*fs)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_range_time, y=time_range_signal, mode='lines', name='Time Range Signal'))
            fig.update_layout(title=f'{selected_channel} Signal from {time_min} to {time_max} seconds',
                              xaxis_title='Time (s)',
                              yaxis_title='Amplitude')
            st.plotly_chart(fig)

            st.subheader(f'Summary Statistics for {selected_channel}')
            st.write(df[selected_channel].describe())
            
            st.sidebar.subheader('Download Filtered Signals')
            download_options = {}
            for band in bands.keys():
                download_options[band] = st.sidebar.checkbox(f'Download {band} Waves')
            
            for band, (low, high) in bands.items():
                if download_options[band]:
                    filtered_signal = bandpass_filter(signal, low, high, fs)
                    if smoothing:
                        filtered_signal = gaussian_filter1d(filtered_signal, sigma=smoothing_sigma)
                    signal_csv = pd.DataFrame({f'{selected_channel}_{band.lower()}': filtered_signal})
                    st.sidebar.download_button(f'Download {band} CSV', signal_csv.to_csv(index=False), f'{selected_channel}_{band.lower()}.csv')

# Image Analysis Page
elif page_selection == "Image Analysis":
    st.title('Image Analysis and Visualization')

    uploaded_image = st.sidebar.file_uploader("Choose an Image", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Add further analysis functionalities for images as required.
        # For instance, you can add image filtering, edge detection, etc.
        st.write("Additional image analysis functionalities can be added here.")

# Instructions for the user
st.write("Use the sidebar to select between EEG signal analysis or image analysis.")
