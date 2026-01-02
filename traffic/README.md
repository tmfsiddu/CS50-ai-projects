# Traffic Sign Classification (CNN)

This project is part of **CS50’s Introduction to Artificial Intelligence with Python (Harvard University)**.  
It focuses on building a **Convolutional Neural Network (CNN)** to classify traffic signs using image data.

---

## Problem Description

The goal of this project is to train a neural network that can accurately identify traffic signs from images.  
This is a supervised learning problem where the model learns to map images to predefined traffic sign categories.

---

## Dataset

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

### Dataset Characteristics
- Real-world traffic sign images
- Multiple classes (traffic sign categories)
- Varying lighting conditions, angles, and resolutions

The dataset is **not included** in this repository due to size constraints.

---

## How to Obtain the Dataset

1. Download the GTSRB dataset from:
   - https://benchmark.ini.rub.de/gtsrb_news.html
2. Extract the dataset
3. Place the dataset directory inside the `traffic/` folder as instructed in the CS50 problem specification

---

## Model Architecture

- Convolutional Neural Network (CNN)
- Implemented using **TensorFlow / Keras**
- Includes:
  - Convolution layers
  - Pooling layers
  - Fully connected layers
  - Softmax output for classification

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV (image preprocessing)

---

## Notes

- This implementation is based on **distribution code provided by CS50**
- All model logic, training pipeline, and modifications were implemented by me as part of coursework
- Dataset files are excluded intentionally

---

## Course Reference

CS50’s Introduction to Artificial Intelligence with Python  
Harvard University
