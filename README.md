# Momenta - Audio Deepfake Detection

Momenta is a deep learning project designed to detect deepfake audio content by analyzing various audio characteristics and features. The system uses a convolutional neural network (CNN) to distinguish between authentic and synthetically generated voice recordings.

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

```bash
HTML
git clone https://github.com/aditisingh02/Momenta.git

SSH
git@github.com:aditisingh02/deepfake-audio-detection.git

cd Momenta
```

2. Install required packages:

```bash
pip install -r requirements.txt
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

### Training New Models

To train a new model on your dataset:

```bash
python train_model_v2.py
```

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

- High accuracy in distinguishing real from fake audio
- Fast inference time
- Robust feature extraction

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Voice samples from various public sources
- Deep learning architecture inspired by state-of-the-art audio processing techniques
