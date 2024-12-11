# Smart Video Detection: Deep Learning for Enhanced Home Security

## 📘 **Project Overview**
The **Smart Video Detection** project leverages deep learning to perform real-time **facial recognition** for **access control** in home security systems. This project utilizes **pre-trained models** such as CNN, VGG16, and ResNet50 to detect and classify faces from video feeds, granting or denying access accordingly.

This system aims to provide an intelligent, secure, and automated solution for recognizing authorized individuals and controlling access based on facial features.

---

## 🌟 **Features**
- **Facial Recognition**: Detect and classify faces from video feeds.
- **Access Control**: Grant or deny access based on recognized faces.
- **Multiple Models**: Uses CNN, VGG16, and ResNet50 for high accuracy.
- **Video Input**: Process video files and live feeds for face detection and classification.
- **Real-time Classification**: Identify and grant access to faces in real-time.

---

## 📂 **Project Structure**
```
📦 Smart Video Detection
 ┣ 📂 dataset       -- Contains images for training the models (family members, etc.)
 ┣ 📂 models        -- Stores trained models (CNN, VGG16, ResNet50)
 ┣ 📂 scripts       -- Python scripts for training and detection
 ┣ 📜 requirements.txt -- Required Python dependencies
 ┗ 📜 README.md    -- Documentation for the project
```

---

## ⚙️ **Setup Instructions**

### **Prerequisites**
Ensure you have the following tools and libraries installed:
- **Python 3.6+**
- **Google Colab** (optional if running in the cloud) or **Local Python Environment** (for local execution)
- **Git** for version control

### **Installing Dependencies**
To install the required dependencies, use the following command:
```bash
pip install -r requirements.txt
```
This will install all necessary Python libraries.

---

## 🚀 **Running the Scripts**

### **1. Training the Models**
You can train the models using the provided scripts. The following commands demonstrate how to train each model:

#### **Train the CNN Model**
```bash
python scripts/train_cnn.py
```
This script will use your dataset to train a CNN-based model.

#### **Train the VGG16 Model**
```bash
python scripts/train_vgg16.py
```

#### **Train the ResNet50 Model**
```bash
python scripts/train_resnet50.py
```
After training, the models will be saved in the **models/** directory.

---

### **2. Face Detection and Access Control**
To process a video feed and classify faces, run the following command:
```bash
python scripts/video_detection.py
```
This script will process the video feed, detect faces, and classify them based on the trained model. You will see the following access control messages on the screen:
- **"ACCESS GRANTED! Welcome Home, [Name]"** for recognized faces.
- **"ACCESS DENIED"** for unrecognized or unknown faces.

> **Note:** Ensure that your dataset is stored in the **dataset/** folder and the trained models are in the **models/** folder.

---

## 📦 **Dataset and Models**

- **Dataset**: The dataset used for training the models should be stored in the **dataset/** folder. The dataset may include labeled images of authorized users (e.g., family members).
- **Models**: Trained models (CNN, VGG16, ResNet50) are saved in the **models/** directory after training. These models are used for face recognition and access control.

---

## 🌐 **How It Works**
1. **Video Feed Processing**: The system takes a video input (live feed or pre-recorded video).
2. **Face Detection**: It identifies and extracts facial regions from the video feed.
3. **Classification**: The facial features are compared to the trained models (CNN, VGG16, ResNet50) to recognize the person.
4. **Access Control**: If the face is recognized, access is granted with a friendly message. If not, access is denied.

---

## 💡 **Future Improvements**
- **Improved Accuracy**: Collect more diverse training data and experiment with hyperparameters.
- **Real-time Detection**: Deploy the system on a Raspberry Pi or an edge device for real-time recognition.
- **Multi-Face Detection**: Enhance the system to handle multiple faces in a single frame.
- **Web Interface**: Create a web-based dashboard for remote monitoring and access control.

---

## 🛠️ **Technologies Used**
- **Python**: Primary language for scripting and machine learning logic.
- **Deep Learning Models**: Pre-trained models (CNN, VGG16, ResNet50) for facial recognition.
- **OpenCV**: Used for face detection in video feeds.
- **TensorFlow / Keras**: For model training and facial classification.

---

## 👨‍💻 **Authors**
- **Hemanthkumar**
- **Dineshkumar**
- **Gokul**

If you have any questions, suggestions, or want to contribute, feel free to reach out!

