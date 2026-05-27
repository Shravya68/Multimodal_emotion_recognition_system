# Multimodal Emotion Recognition

A deep learning-based multimodal emotion recognition system using speech, text, and fusion-based learning on the TESS (Toronto Emotional Speech Set) dataset.

---

## Project Overview

This project implements three emotion recognition pipelines:

- Speech Emotion Recognition
- Text Emotion Recognition
- Multimodal Fusion-based Emotion Recognition

The system analyses emotional information from audio signals and textual input and combines both modalities using weighted decision-level fusion.

---

## Dataset

Dataset Used:
- TESS (Toronto Emotional Speech Set)

Dataset Details:
- 2800 audio samples
- 7 emotion classes
- Two female speakers (OAF and YAF)

### Emotion Classes
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Pleasant Surprise (PS)
- Sad

---

## Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## Project Structure

```text
## Project Structure

```text
multimodal_emotion_recognition/
│
├── models/
│   ├── speech_pipeline/
│   │   ├── speech_processing.py
│   │   ├── speech_train.py
│   │   ├── speech_test.py
│   │   └── notebooks/
│   │
│   ├── text_pipeline/
│   │   ├── text_train.py
│   │   ├── text_test.py
│   │   └── notebooks/
│   │
│   └── fusion_pipeline/
│       ├── fusion_train.py
│       ├── fusion_test.py
│       └── notebooks/
│
├── Results/
│   ├── accuracy_tables.pdf
│   └── plots/
│
├── report.pdf
├── requirements.txt
└── README.md
```
```
```

---

## Speech Pipeline

The speech emotion recognition pipeline includes:

- Audio preprocessing
- Silence trimming
- MFCC feature extraction
- Conv1D + Bidirectional LSTM model

### Speech Model Accuracy
- 99.82%

---

## Text Pipeline

The text emotion recognition pipeline includes:

- Word extraction from filenames
- Tokenization
- Embedding layer
- Bidirectional GRU model

### Text Model Accuracy
- 12.86%

The lower accuracy is expected because TESS contains emotionally neutral words repeated across all emotion classes.

---

## Fusion Pipeline

The multimodal fusion pipeline combines speech and text predictions using weighted decision-level fusion.

### Fusion Formula

```text
Final Probability =
(0.7 × Speech Probability) +
(0.3 × Text Probability)
```

### Fusion Model Accuracy
- 99.93%

---

## Results Summary

| Model | Test Accuracy |
|---|---|
| Speech-only | 99.82% |
| Text-only | 12.86% |
| Fusion (0.7 / 0.3) | 99.93% |

---

## Setup Instructions

Clone the repository:

```bash
git clone https://github.com/yourusername/multimodal-emotion-recognition.git
```

Move into the project directory:

```bash
cd multimodal-emotion-recognition
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Speech Pipeline

Speech Processing:
```bash
python speech_processing.py
```

Train:
```bash
python speech_train.py
```

Test:
```bash
python speech_test.py
```

### Text Pipeline

Train:
```bash
python text_train.py
```

Test:
```bash
python text_test.py
```

### Fusion Pipeline

Train:
```bash
python fusion_train.py
```

Test:
```bash
python fusion_test.py
```

---

## Key Observations

- Speech carries strong emotional information in the TESS dataset.
- Text-only emotion recognition performs poorly because the dataset words are emotionally neutral.
- Fusion performance closely matches speech performance because speech dominates emotional representation in TESS.

---
