# Audio Deepfake Detection

This is a deep learning project designed to detect deepfake audio content by analyzing various audio characteristics and features. The system uses a convolutional neural network (CNN) to distinguish between authentic and synthetically generated voice recordings.

https://github.com/user-attachments/assets/0fb44f11-8717-43f5-81c4-27e68b14df8a

## Dataset

The project uses the Deep Voice Deepfake Voice Recognition dataset from Kaggle:

- Dataset Link: [Deep Voice Deepfake Voice Recognition](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition/data)
- Contains real and synthetic voice samples
- Used for training and testing the model

## Features

- Real-time audio deepfake detection
- Advanced feature extraction including:
  - Mel spectrograms
  - MFCCs (Mel-frequency cepstral coefficients)
  - Spectral rolloff
  - Zero crossing rate
- Visual analysis through spectrogram plotting
- Support for various audio formats
- Pre-trained model included (`best_model_v2.h5`)

## Dataset Structure

The project uses a structured dataset organized as follows:

```
data/
├── AUDIO/
│   ├── REAL/          # Original voice recordings
│   │   └── *.wav      # Real audio samples
│   └── FAKE/          # Synthetic voice recordings
│       └── *.wav      # Generated audio samples
└── DATASET-balanced.csv   # Metadata and labels
```

## Requirements

- Python 3.11 or higher
- TensorFlow 2.15.0
- librosa 0.10.1
- numpy 1.24.3
- pandas 2.1.4
- scikit-learn 1.3.2
- matplotlib 3.8.2
- soundfile 0.12.1

## Installation

1. Clone the repository:

### HTML

```bash
git clone https://github.com/aditisingh02/Momenta.git
```

### SSH

```bash
git@github.com:aditisingh02/deepfake-audio-detection.git
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Download the dataset and follow the file structure

## Project Structure

```
deepfake-audio-detection/
├── app.py                 # main application
├── train_model_v2.py      # Model training script
├── test_model.py          # Model testing script
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python dependencies
├── .streamlit/
│   └── config.toml       # theme and app settings
├── deepfake_audio_detector_v2.h5
└── best_model_v2.h5      # final model
```

### Training the Model

To train a new model on your dataset:

```bash
python train_model_v2.py
```

### Run the Application

```bash
python -m streamlit run app.py
```

## Usage

### Testing Audio Files

To test an audio file for authenticity:

```bash
python test_model.py --model best_model_v2.h5 --audio "path/to/audio/file.wav"
```

Example:

```bash
python test_model.py --model best_model_v2.h5 --audio "data/AUDIO/REAL/biden-original.wav"
```

### Running the application

1. Launch the application using `python -m streamlit run app.py`
2. Upload a WAV audio file using the file uploader
3. Wait for the analysis to complete
4. View the detection results, confidence score, and audio visualizations

The training script includes:

- Data augmentation for better generalization
- Early stopping to prevent overfitting
- Model checkpointing to save the best version
- Learning rate reduction on plateau

## Model Architecture

The CNN model architecture includes:

- 3 convolutional layers (32, 64, 128 filters)
- Batch normalization
- Max pooling layers
- Dropout for regularization
- Dense layers for classification

## Performance

The model achieves:

- Test Accuracy: ~90.91% (based on validation results)
- AUC Score: High discriminative ability between real and fake audio
- Fast inference time (typically under 1 second per audio sample)
- Robust feature extraction with multiple audio characteristics:
  - Mel spectrograms
  - MFCCs
  - Spectral rolloff
  - Zero crossing rate

## Acknowledgments

- Voice samples from [Deep Voice Deepfake Voice Recognition](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition/data)
