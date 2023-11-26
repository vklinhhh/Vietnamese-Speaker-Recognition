import streamlit as st
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from scipy.io.wavfile import read as wav_read
from IPython.display import Audio
import tempfile
import os
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import glob


# Map hình ảnh
image_folder = "./asset/photo/"
name_df = pd.read_csv("./asset/name.csv")
name_mapping = dict(zip(name_df["folder_name"], name_df["new_name"]))

### Khởi tạo model
SAMPLING_RATE = 16000

model_path = "./model/cnn_model_v2.h5"
model = load_model(model_path)

def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio

def audio_to_fft(audio):
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

class_names = ['VIVOSSPK01', 'VIVOSSPK02', 'VIVOSSPK03', 'VIVOSSPK04', 'VIVOSSPK05', 'VIVOSSPK06',
            'VIVOSSPK07', 'VIVOSSPK08', 'VIVOSSPK09', 'VIVOSSPK10', 'VIVOSSPK11', 'VIVOSSPK12',
            'VIVOSSPK13', 'VIVOSSPK14', 'VIVOSSPK15', 'VIVOSSPK16', 'VIVOSSPK17', 'VIVOSSPK18',
            'VIVOSSPK19', 'VIVOSSPK20', 'VIVOSSPK21', 'VIVOSSPK22', 'VIVOSSPK23', 'VIVOSSPK24',
            'VIVOSSPK25', 'VIVOSSPK26', 'VIVOSSPK27', 'VIVOSSPK28', 'VIVOSSPK29', 'VIVOSSPK30',
            'VIVOSSPK31', 'VIVOSSPK32', 'VIVOSSPK33', 'VIVOSSPK34', 'VIVOSSPK35', 'VIVOSSPK36',
            'VIVOSSPK37', 'VIVOSSPK38', 'VIVOSSPK39', 'VIVOSSPK40', 'VIVOSSPK41', 'VIVOSSPK42',
            'VIVOSSPK43', 'VIVOSSPK44', 'VIVOSSPK45', 'VIVOSSPK46']


# Streamlit app
def main():
    ##set background
    st.set_page_config(layout="wide")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://wallpaperaccess.com/full/3192384.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    css = f"""
    <style>
        .centered-text {{
            text-align: center;
            color: grey;
            background-image: url("https://media.tenor.com/BOu8ryjIR38AAAAC/sound-wave-wave.gif");
            background-repeat: repeat;
            background-size: contain;
            padding: 20px;
        }}
    </style>
        """
    st.markdown(css,unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #f4e409;' class='centered-text'>Vietnamese Speaker Recognition</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #faedcd;'>Author: Linh Vo</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #fefae0;'>The Vietnamese Speaker Recognition Application is a tool designed to identify and determine speakers within Vietnamese audio files.</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #fefae0;'>Using deep learning models, this application can recognize and classify speakers based on their unique audio features.</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #fefae0;'>With the ability to fine-tune and accurately identify speakers, this application can be applied across various domains, from security to voice recordings and user authentication.</h3>", unsafe_allow_html=True)
    # File upload
    new_audio_file_path = st.file_uploader("Upload a WAV file", type=["wav"])

    if new_audio_file_path is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
            temp_wav.write(new_audio_file_path.read())

        # Đọc WAV file
        wav_data = wav_read(temp_wav_path)[1]
        audio = np.array(wav_data, dtype=np.float32)

        # Preprocess the audio
        preprocessed_audio = path_to_audio(temp_wav_path)
        preprocessed_audio = tf.expand_dims(preprocessed_audio, axis=0)
        new_fft = audio_to_fft(preprocessed_audio)

        # Predict the speaker
        y_pred = model.predict(new_fft)
        predicted_label = np.argmax(y_pred, axis=-1)
        predicted_speaker = class_names[predicted_label[0]]
        speaker_name = name_mapping[predicted_speaker]
        # Display the predicted speaker
        predicted_speaker_name = predicted_speaker.replace(" ", "")  # Remove spaces from the name
        image_filenames = glob.glob(image_folder + predicted_speaker_name + "*.png")
        
        # Display the audio waveform
        st.write("Audio waveform:")
        st.audio(temp_wav_path)
        os.remove(temp_wav_path)

        # Tạo waveform và spectrogram
        fig, (ax_waveform, ax_spectrogram) = plt.subplots(2, 1, figsize=(15, 4))
        ax_waveform.set_facecolor('none')
        ax_spectrogram.set_facecolor('none')
        librosa.display.waveshow(audio, sr=SAMPLING_RATE, ax=ax_waveform, alpha=0.5, color='#eeba0b')
        ax_waveform.set_title("Waveform", fontname='Arial', fontsize=8)
        ax_waveform.set_xlabel("Time (s)", color="white", fontname='Arial', fontsize=8)
        ax_waveform.set_ylabel("Amplitude", color="white" , fontname='Arial', fontsize=8)
        ax_waveform.tick_params(axis='x', colors='white', labelsize=6)
        ax_waveform.tick_params(axis='y', colors='white', labelsize=6)
        fig.patch.set_alpha(0.0)
        for spine in ax_waveform.spines.values():
            spine.set_edgecolor('white')
        ax_waveform.title.set_color('white')

        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, sr=SAMPLING_RATE, x_axis='time', y_axis='log', ax=ax_spectrogram, cmap='afmhot')
        ax_spectrogram.set_title("Spectrogram", color="white", fontsize=8, fontname='Arial')
        ax_spectrogram.set_xlabel("Time", color="white" , fontsize=8, fontname='Arial')
        ax_spectrogram.set_ylabel("Frequency", color="white" , fontsize=8, fontname='Arial')
        ax_spectrogram.tick_params(axis='x', colors='white', labelsize=6)
        ax_spectrogram.tick_params(axis='y', colors='white', labelsize=6)
        col1,col2,col3,col4 = st.columns([0.05,0.15, 0.75,0.05])
        with col2:
            st.markdown(f'<p style="text-align:center; color:white;">Predicted speaker: {speaker_name}</p>', unsafe_allow_html=True)
            if image_filenames:
                image_path = image_filenames[0]
                st.image(image_path, use_column_width="always")
            else:
                st.write("No image found for " + predicted_speaker)
        with col3:        
            st.pyplot(fig)
if __name__ == "__main__":
    main()