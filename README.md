# Image Classification App with Streamlit

This repository contains a Streamlit-based app for real-time image classification. Users can dynamically upload and load models in `.zip` format and classify images from a webcam or uploaded image files.

## Features

1. **Dynamic Model Loading**:
   - Upload a `.zip` file containing the model (`.h5` file) and `labels.txt`.
   - The app extracts and loads the model dynamically.

2. **Modes for Input**:
   - Webcam: Capture images for real-time classification.
   - Upload Image: Classify images from local files.

3. **Interactive and Easy-to-Use**:
   - Streamlit provides an intuitive web interface for interaction.

## Repository Structure
# Project Directory Structure

```plaintext
ImageClassificationApp/
├── app.py                  # Main Streamlit app
├── models/                 # Folder to store model zip files and extracted contents
│   ├── sample_model.zip    # Example model zip file (with .h5 and labels.txt inside)
├── README.md               # Instructions and project details
└── requirements.txt        # Python dependencies
```



## Prerequisites

1. Python 3.7 or higher.
2. Install required Python libraries:
   ```bash
   pip install -r requirements.txt
