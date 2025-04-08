import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import random

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

def load_and_preprocess_audio(file_path, augment=False):
    """Load and preprocess audio file with optional augmentation"""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(audio) < SAMPLES:
            audio = np.pad(audio, (0, SAMPLES - len(audio)))
        
        if augment:
            #Random time shift
            shift = int(random.uniform(-0.1, 0.1) * len(audio))
            audio = np.roll(audio, shift)
            
            #Random pitch shift
            n_steps = random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)
            
            #Random volume change
            audio = audio * random.uniform(0.8, 1.2)
        
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_features(audio):
    """Extract audio features focusing on voice characteristics"""
    features = []
    
    #Mel spectrogram with voice-focused parameters
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=SAMPLE_RATE,
        n_mels=128,
        hop_length=512,
        n_fft=2048,
        fmin=20,
        fmax=8000  #to focus on the human voice frequency range
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.append(mel_spec_db)
    
    #MFCC 
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=20)
    features.append(mfcc)
    
    #Spectral Rolloff to detect voice authenticity
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE).reshape(1, -1)
    features.append(rolloff)
    
    #Zero Crossing Rate - to detect synthetic artifacts
    zcr = librosa.feature.zero_crossing_rate(audio).reshape(1, -1)
    features.append(zcr)
    
    return np.concatenate(features, axis=0)

def create_model(input_shape):
    """Create a more focused model for voice deepfake detection"""
    model = models.Sequential([
        #Input layer with stronger regularization
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape,
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        #Mid-level feature extraction
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        #High-level feature extraction
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        #Dense layers for classification
        layers.Flatten(),
        layers.Dense(256, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    #lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

def main():
    print("Loading dataset...")
    
    X = []
    y = []
    
    #Process real audio files
    real_dir = "data/AUDIO/REAL"
    for file in os.listdir(real_dir):
        if file.endswith('.wav'):
            print(f"Processing real audio: {file}")
            #original audio
            audio = load_and_preprocess_audio(os.path.join(real_dir, file))
            if audio is not None:
                features = extract_features(audio)
                X.append(features)
                y.append(0)
            
            #create 2 augmented versions
            for i in range(2):  
                audio = load_and_preprocess_audio(os.path.join(real_dir, file), augment=True)
                if audio is not None:
                    features = extract_features(audio)
                    X.append(features)
                    y.append(0)
    
    #process the fake audio files
    fake_dir = "data/AUDIO/FAKE"
    for file in os.listdir(fake_dir):
        if file.endswith('.wav'):
            print(f"Processing fake audio: {file}")
            audio = load_and_preprocess_audio(os.path.join(fake_dir, file))
            if audio is not None:
                features = extract_features(audio)
                X.append(features)
                y.append(1)
    
    X = np.array(X)
    y = np.array(y)
    
    # add channel dimension
    X = np.expand_dims(X, axis=-1)
    
    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    #Create and train model
    model = create_model(X_train.shape[1:])
    model.summary()
    
    #model checkpoint
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model_v2.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    #train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    #Evaluate model
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    #Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_v2.png')
    print("\nTraining history plot saved as 'training_history_v2.png'")
    
    #save the final model
    model.save('deepfake_audio_detector_v2.h5')
    print("Model saved as 'deepfake_audio_detector_v2.h5'")

if __name__ == "__main__":
    main() 