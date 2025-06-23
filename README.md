## 🧠 Simple Hand Gesture Recognition System (Using OpenCV + Random Forest)

This mini project implements a **gesture recognition system** that can identify basic hand gestures (like Open Hand, Thumbs Up, etc.) using a webcam and a machine learning model trained on contour features of the hand.

### ✨ Features

* Real-time hand detection using background subtraction (no MediaPipe required!)
* Feature extraction from hand contour: area, perimeter, convexity defects, Hu moments, and more
* Classification using Random Forest
* Easy-to-use console interface:

  * Collect Training Data
  * Train Model
  * Real-time Recognition

### 🔧 Requirements

* Python 3.7+
* OpenCV (`cv2`)
* NumPy
* scikit-learn
* pickle

You can install the dependencies using:

```bash
pip install opencv-python numpy scikit-learn
```

### 🚀 How to Use

1. Run the script:

   ```bash
   python gesture_recognizer.py
   ```

2. Choose one of the options from the menu:

   * `1`: Collect training data for predefined gestures
   * `2`: Train a Random Forest model on the collected data
   * `3`: Start real-time gesture recognition using webcam
   * `4`: Load previously saved model
   * `5`: Exit the program

### 📂 Data Files

* `gesture_data_simple.pickle`: Stores training features and labels
* `gesture_model_simple.pickle`: Serialized trained model

> These files are ignored from version control via `.gitignore`

### 💡 Notes

* Ensure good lighting and a contrasting background during training.
* The system assumes the **largest contour** in the foreground is the hand.
* You can easily extend this to support more gestures by updating the `gesture_labels` list.

---

### 📸 Sample Gestures (Recommended)

* Open Hand ✋
* Closed Fist ✊
* Peace Sign ✌️
* Thumbs Up 👍

---

### 📜 License

This project is open-source and free to use for learning and experimentation.
