# My-New-App
import streamlit as st
from PIL import Image
import io
import cv2
import numpy as np

st.title("Image Input with Face Detection")

# Create a radio button for image input options
option = st.radio("Choose an option:", ("Browse Image", "Capture Image"))

# Function to perform face detection
def detect_faces(image):
    # Convert PIL Image to OpenCV format (numpy array)
    image_np = np.array(image)

    # Convert RGB to BGR (OpenCV uses BGR format)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Load a pre-trained face detection model (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert back to RGB for Streamlit display
    image_with_faces = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return image_with_faces, len(faces)

if option == "Browse Image":
    # Upload image from local system
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert uploaded file to PIL Image
        image = Image.open(uploaded_file)

        # Perform face detection
        image_with_faces, face_count = detect_faces(image)

        # Display the uploaded and processed images
        st.image(image, caption="Uploaded Image")
        st.image(image_with_faces, caption="Image with Detected Faces", use_column_width=True)

        # Show the number of detected faces
        st.write(f"Detected {face_count} face(s) in the image.")

elif option == "Capture Image":
    # Capture image using the camera
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # Convert image buffer to bytes and then to PIL Image
        bytes_data = img_file_buffer.getvalue()
        image = Image.open(io.BytesIO(bytes_data))

        # Perform face detection
        image_with_faces, face_count = detect_faces(image)

        # Display the captured and processed images
        st.image(image, caption="Captured Image", use_column_width=True)
        st.image(image_with_faces, caption="Image with Detected Faces", use_column_width=True)

        # Show the number of detected faces
        st.write(f"Detected {face_count} face(s) in the image.")
