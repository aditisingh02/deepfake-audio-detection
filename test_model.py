import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import time

# Constants
SAMPLE_RATE = 22050
DURATION = 3  # seconds
SAMPLES = SAMPLE_RATE * DURATION

def load_audio(file_path):
    """Load and preprocess audio file"""
    try:
        print(f"Loading audio file: {file_path}")
        start_time = time.time()
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(audio) < SAMPLES:
            audio = np.pad(audio, (0, SAMPLES - len(audio)))
        print(f"Audio loaded in {time.time() - start_time:.2f} seconds")
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_features(audio):
    """Extract audio features focusing on voice characteristics"""
    features = []
    
    # Mel spectrogram with voice-focused parameters
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=SAMPLE_RATE,
        n_mels=128,
        hop_length=512,
        n_fft=2048,
        fmin=20,
        fmax=8000  # Focus on human voice frequency range
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.append(mel_spec_db)
    
    # MFCC (good for voice characteristics)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=20)
    features.append(mfcc)
    
    # Spectral Rolloff (helps detect voice authenticity)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE).reshape(1, -1)
    features.append(rolloff)
    
    # Zero Crossing Rate (helps detect synthetic artifacts)
    zcr = librosa.feature.zero_crossing_rate(audio).reshape(1, -1)
    features.append(zcr)
    
    return np.concatenate(features, axis=0)

def plot_spectrogram(spec, title):
    """Plot mel spectrogram"""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec, sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def test_audio(model, file_path):
    """Test a single audio file"""
    print(f"\nTesting audio file: {file_path}")
    
    # Load and process audio
    audio = load_audio(file_path)
    if audio is None:
        return
    
    # Extract features
    features = extract_features(audio)
    
    # Plot original spectrogram
    print("Plotting spectrogram...")
    plot_spectrogram(features[:128], "Mel Spectrogram")  # Only plot mel spectrogram
    
    # Prepare features for prediction
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    features = np.expand_dims(features, axis=-1)  # Add channel dimension
    
    # Make prediction
    print("Making prediction...")
    start_time = time.time()
    try:
        prediction = model.predict(features, verbose=1)[0][0]
        print(f"Prediction completed in {time.time() - start_time:.2f} seconds")
        
        confidence = prediction if prediction > 0.5 else 1 - prediction
        label = "Fake" if prediction > 0.5 else "Real"
        
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Raw prediction value: {prediction:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

def main():
    parser = argparse.ArgumentParser(description='Test Deepfake Audio Detection Model')
    parser.add_argument('--model', type=str, default='deepfake_audio_detector.h5',
                        help='Path to the trained model file')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to the audio file to test')
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.model}...")
    start_time = time.time()
    model = tf.keras.models.load_model(args.model)
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Print model summary
    model.summary()
    
    # Test the audio file
    test_audio(model, args.audio)

if __name__ == "__main__":
    main() 