# Human Activity Recognition (HAR) using CNN and Smartphone Accelerometer Data

This project focuses on Human Activity Recognition (HAR) using time-series data collected from a 3-axis accelerometer. The data represents six physical activities — walking, walking upstairs, walking downstairs, sitting, standing, and laying — performed by volunteers wearing smartphones. We use a 1D Convolutional Neural Network (CNN) to classify these activities based on temporal patterns in the sensor data.

The dataset has been preprocessed into fixed-size windows (128 samples per window) with a sampling frequency of 50Hz. After loading and preprocessing, the model is trained and validated using a typical 85-15 split and evaluated on a held-out test set.

To ensure edge compatibility, the trained Keras model is converted to TensorFlow Lite (TFLite) format, followed by post-training quantization for optimized deployment on low-power devices such as the ESP32 microcontroller. The quantized model is also exported as a C header file for direct integration into embedded firmware.

The project includes full evaluation metrics, such as confusion matrix, accuracy, F1 scores, and inference timing for both TFLite and quantized versions. This pipeline demonstrates an end-to-end workflow from model training to embedded deployment, making it suitable for real-time activity recognition applications on resource-constrained devices.


## 📁 Dataset

The original dataset is the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones), collected from 30 volunteers using a Samsung Galaxy S II smartphone.

**Note**: Preprocessed CSV versions (`accelerometer_3axis_time_series_train.csv`, etc.) must be placed in the appropriate folder (see `src/train_model.py`).

#  📂 Dataset Structure Overview 

# 📁 CSV Files Used:
 ----------------------------------------------------------------------------------
 1. accelerometer_3axis_train.csv   → Shape: (7352, 128, 3)
 2. accelerometer_3axis_test.csv    → Shape: (2947, 128, 3)
 3. y_train_labels.csv              → Shape: (7352, 6)   # One-hot encoded labels
 4. y_test_labels.csv               → Shape: (2947, 6)   # One-hot encoded labels

# 🧠 Data Dimensions:
 ----------------------------------------------------------------------------------
 - Each sample = 128 time steps (over 2.56 seconds) of 3-axis accelerometer data
 - Shape per sample = (128, 3) → [X, Y, Z] acceleration readings
 - Final shape for training:   X_train → (6249, 128, 3),  y_train → (6249, 6)
 - Final shape for validation: X_val   → (1103, 128, 3),  y_val   → (1103, 6)
 - Final shape for testing:    X_test  → (2947, 128, 3),  y_test  → (2947, 6)

# 📌 Class Labels (6 total):
 ----------------------------------------------------------------------------------
 0 → WALKING
 1 → WALKING_UPSTAIRS
 2 → WALKING_DOWNSTAIRS
 3 → SITTING
 4 → STANDING
 5 → LAYING

# 📝 Notes:
 - The CSVs were generated from UCI HAR Dataset inertial signal files.
 - Values are normalized and suitable for training with 1D CNN models.
 - Data split: 85% train / 15% validation from original training set.

# ===========================================================================================

## 🧠 Model Architecture

A 1D Convolutional Neural Network (CNN) is used:
- Input shape: (128, 3) — 3-axis accelerometer data in time windows
- Conv1D → MaxPool → Conv1D → MaxPool → Flatten → Dense → Dropout → Output
- Output: Softmax layer for 6-class classification

## 📊 Results

| Metric           | Value |
|------------------|-------|
| Validation Acc   | ~X.XX |
| Test Accuracy    | ~X.XX |
| Macro F1-Score   | ~X.XX |
| Micro F1-Score   | ~X.XX |

## 🛠️ Files & Directories

- `src/`: Python scripts for training, evaluation, model export
- `models/`: Trained Keras, TFLite, and C header model
- `data/`: Dataset instructions or metadata
- `notebooks/`: Optional explorations
- `requirements.txt`: Required Python packages

## 📦 Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
