# 👗 Fashion MNIST Classification

A deep learning project that classifies images of fashion items such as shirts, shoes, and dresses from the **Fashion MNIST** dataset using **TensorFlow/Keras**.

---

## 📌 Project Overview

This project aims to build and train a **Convolutional Neural Network (CNN)** capable of recognizing different clothing categories from grayscale images.  
The model achieves high accuracy through effective data preprocessing, model tuning, and visualization of results.

---

## 🧠 Dataset Information

The **Fashion MNIST** dataset consists of:
- **60,000 training images**
- **10,000 test images**
- **10 clothing categories**, including:

| Label | Description     |
|:------:|:----------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Each image is **28x28 pixels**, grayscale, centered on the item.

---

## ⚙️ Model Architecture

The CNN model includes:
- **Convolutional layers** for feature extraction  
- **MaxPooling layers** to reduce spatial dimensions  
- **Dropout** to prevent overfitting  
- **Dense layers** for final classification  

## 🚀 How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Manwikkk/fashion-MNIST-classification.git
cd fashion-MNIST-classification
```
