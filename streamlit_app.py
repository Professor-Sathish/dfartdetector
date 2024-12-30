import streamlit as st
import zipfile
import os
import cv2
from keras.models import load_model
import numpy as np
from PIL import Image

# Streamlit app title
st.title("Dynamic Model Selection and Image Classification")

# Sidebar for model selection
st.sidebar.title("Model Selection")
uploaded_model = st.sidebar.file_uploader("Upload a Model (ZIP File)", type="zip")

# Directory to extract models
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Placeholder for selected model
selected_model_path = None

if uploaded_model:
    with zipfile.ZipFile(uploaded_model, "r") as zip_ref:
        zip_ref.extractall(model_dir)
    st.sidebar.success("Model extracted successfully!")
    
    # Find the model file inside the extracted folder
    extracted_files = os.listdir(model_dir)
    for file in extracted_files:
        if file.endswith(".h5"):
            selected_model_path = os.path.join(model_dir, file)
            break

# Load the model if available
model = None
if selected_model_path:
    st.sidebar.info(f"Loading model: {os.path.basename(selected_model_path)}")
    model = load_model(selected_model_path, compile=False)
    st.sidebar.success("Model loaded successfully!")

# Load the labels
labels_file = os.path.join(model_dir, "labels.txt")
class_names = []
if os.path.exists(labels_file):
    with open(labels_file, "r") as file:
        class_names = file.readlines()
else:
    st.sidebar.warning("Labels file not found in the model zip!")

# Choose input method
st.sidebar.title("Choose Input Method")
mode = st.sidebar.radio("Select Input:", ("Webcam", "Upload Image"))

if model:
    if mode == "Webcam":
        st.markdown("### Webcam Feed")
        run = st.checkbox("Run Webcam")
        FRAME_WINDOW = st.image([])

        # Initialize webcam
        camera = cv2.VideoCapture(0)

        while run:
            # Read frame from the webcam
            ret, frame = camera.read()
            if not ret:
                st.error("Error accessing webcam")
                break
            
            # Display the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb, channels="RGB")

            # Capture and process frame for prediction
            if st.button("Capture and Classify"):
                image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
                image = (image / 127.5) - 1
                
                # Predict the class
                prediction = model.predict(image)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]

                # Display results
                st.success(f"**Class:** {class_name.strip()} | **Confidence Score:** {confidence_score * 100:.2f}%")
                break

        # Release webcam after exiting
        camera.release()

    elif mode == "Upload Image":
        st.markdown("### Upload Image for Classification")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess the image for the model
            image = image.resize((224, 224))
            image_array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            image_array = (image_array / 127.5) - 1

            # Predict the class
            prediction = model.predict(image_array)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Display results
            st.success(f"**Class:** {class_name.strip()} | **Confidence Score:** {confidence_score * 100:.2f}%")

else:
    st.warning("Please upload and select a model to proceed.")
