import cv2
import numpy as np
from tensorflow.keras.models import load_model
from google.colab.patches import cv2_imshow  # Import cv2_imshow for Colab
import matplotlib.pyplot as plt

# Load the trained model without compiling (to avoid the warning)
model = load_model('/content/drive/MyDrive/models/cnn_family_member_model.h5', compile=False)  # Or any of your models
class_indices = {'Gokul': 0, 'Dineshkumar': 1, 'Hemanth': 2}  # Update based on your classes

def preprocess_image(img):
    img = cv2.resize(img, (128, 128))  # Resize to the model's expected input size
    img_array = np.expand_dims(img, axis=0)
    img_array = img_array.astype('float32') / 255.0  # Normalize the image
    return img_array

def predict_class(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    return list(class_indices.keys())[predicted_class_index], predictions[0][predicted_class_index]

# Function to detect faces using OpenCV
def detect_faces(video_path):
    # Load pre-trained face detection model (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the video
    video = cv2.VideoCapture(video_path)

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop through the faces found
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Extract the face region
            face = frame[y:y+h, x:x+w]

            # Predict the class of the face
            predicted_class, confidence = predict_class(face)

            # Display the class and confidence on the video feed
            if confidence > 0.7:  # You can adjust the threshold for confidence (0.7 is an example)
                message = f"ACCESS GRANTED! Welcome Home, {predicted_class}!"
            else:
                message = "ACCESS DENIED. Sorry, you are not recognized."

            # Display the message on the video feed
            cv2.putText(frame, message, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the resulting frame in Colab
        cv2_imshow(frame)  # Use cv2_imshow for Colab

    video.release()
    cv2.destroyAllWindows()
video_path = '/content/drive/MyDrive/dataset/test_video.mp4'  # Path to your test video
detect_faces(video_path)