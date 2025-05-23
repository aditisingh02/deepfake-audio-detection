import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import os
from PIL import Image
import io
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon=Image.open("public/equalizer.png"),
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Constants
SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

def load_and_preprocess_audio(audio_file):
    """Load and preprocess audio file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name

        audio, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(audio) < SAMPLES:
            audio = np.pad(audio, (0, SAMPLES - len(audio)))
        
        os.unlink(tmp_path)
        return audio
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def extract_features(audio):
    """Extract audio features"""
    features = []
    
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=SAMPLE_RATE,
        n_mels=128,
        hop_length=512,
        n_fft=2048,
        fmin=20,
        fmax=8000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.append(mel_spec_db)
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=20)
    features.append(mfcc)
    
    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE).reshape(1, -1)
    features.append(rolloff)
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio).reshape(1, -1)
    features.append(zcr)
    
    return np.concatenate(features, axis=0)

def plot_audio_features(audio):
    """Plot audio features"""
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Waveform
    librosa.display.waveshow(audio, sr=SAMPLE_RATE, ax=ax1, color='#2DD4BF')
    ax1.set_title('Waveform', color='#F8FAFC', fontsize=14, pad=20)
    ax1.set_xlabel('Time (s)', color='#CBD5E1', fontsize=12)
    ax1.set_ylabel('Amplitude', color='#CBD5E1', fontsize=12)
    ax1.grid(True, alpha=0.2)
    
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, sr=SAMPLE_RATE, ax=ax2, cmap='viridis')
    ax2.set_title('Mel Spectrogram', color='#F8FAFC', fontsize=14, pad=20)
    ax2.set_xlabel('Time (s)', color='#CBD5E1', fontsize=12)
    ax2.set_ylabel('Frequency (Hz)', color='#CBD5E1', fontsize=12)
    ax2.grid(True, alpha=0.2)
    
    plt.colorbar(img, ax=ax2, format='%+2.0f dB', label='dB')
    plt.tight_layout()
    
    return fig

def main():
    # Header
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: 1rem;'>
            Deepfake Audio Detector
        </h1>
        <p style='text-align: center; margin-bottom: 3rem;'>
            Upload your audio file to detect if it's authentic or AI-generated using advanced machine learning
        </p>
    """, unsafe_allow_html=True)

    # File upload
    with st.container():
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3'],
            help="Upload a WAV or MP3 file to analyze"
        )

    if uploaded_file is not None:
        audio = load_and_preprocess_audio(uploaded_file)
        if audio is not None:
            features = extract_features(audio)
            features = np.expand_dims(features, axis=(0, -1))
            
            try:
                model = tf.keras.models.load_model('best_model_v2.h5')
                prediction = model.predict(features)[0][0]
                
                st.markdown("### Detection Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Confidence Score")
                    prediction_float = float(prediction)
                    st.progress(prediction_float)
                    st.markdown(f"**{prediction_float*100:.1f}%** confidence of being AI-generated")
                
                with col2:
                    st.markdown("#### Analysis")
                    if prediction_float < 0.3:
                        st.success("REAL \nThe audio appears to be authentic with natural voice characteristics.")
                    elif prediction_float < 0.7:
                        st.warning("The result is uncertain. The audio shows mixed characteristics.")
                    else:
                        st.error("FAKE \nThe audio shows patterns consistent with AI-generated content.")
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")

            st.markdown("### Audio Analysis")
            fig = plot_audio_features(audio)
            st.pyplot(fig)

if __name__ == "__main__":
    main() 