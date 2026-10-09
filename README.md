# Gesture Detection

A real-time hand gesture detection project using deep learning and computer vision. The system detects hand gestures through a webcam and uses trained models to recognize gestures and perform corresponding actions.

## Features

* Real-time hand gesture detection
* Webcam-based input
* LSTM-based gesture recognition
* Trained deep learning models
* Gesture-based action control
* Real-time prediction

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* MediaPipe
* LSTM

## Project Structure

```text
gesture_detection/
│
├── data/
├── gesture_data/
├── collect_data.py
├── train_model.py
├── train_lstm.py
├── realtime_lstm.py
├── live_detection.py
├── gui_simulation.py
├── train_model.py
├── class_labels.json
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/PiyushRaj282/gesture_detection.git
cd gesture_detection
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

To collect gesture data:

```bash
python collect_data.py
```

To train the model:

```bash
python train_model.py
```

To run real-time gesture detection:

```bash
python live_detection.py
```

## Project Workflow

```text
Webcam
   ↓
Hand Detection
   ↓
Feature Extraction
   ↓
LSTM Model
   ↓
Gesture Prediction
   ↓
Action / Output
```

## Applications

* Human-computer interaction
* Touchless controls
* Sign and gesture recognition
* Smart device control
* Computer vision applications


GitHub: https://github.com/PiyushRaj282
