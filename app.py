# File: streamlit_number_plate_alarm.py

import streamlit as st
import cv2
import numpy as np
import easyocr
from playsound import playsound
from PIL import Image

# Define valid number plates
VALID_NUMBER_PLATES = ["HR.26 BR 9044", "21 BH 2345 AAI", "LMN9876", "DEF5432"]

# Function to process the uploaded image and validate number plate
def process_image(uploaded_image):
    try:
        # Convert uploaded file to raw bytes and decode as OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_image.getvalue()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image. Please upload a valid image file.")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Noise reduction
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

        # Edge detection
        edged = cv2.Canny(bfilter, 30, 200)

        # Find contours
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(keypoints[0], key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:  # If the contour has 4 corners
                location = approx
                break

        if location is None:
            return img, None, "No number plate detected"

        # Create a mask and crop the detected area
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [location], 0, 255, -1)
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        # OCR to extract text
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)

        if not result:
            return img, None, "No text recognized"

        detected_text = result[0][-2].strip()

        # Check if the detected text matches the valid plates
        if detected_text in VALID_NUMBER_PLATES:
            return img, detected_text, "Valid Number Plate"
        else:
            # Draw a red rectangle if invalid
            cv2.drawContours(img, [location], 0, (0, 0, 255), 3)
            playsound("mixkit-classic-alarm-995.wav")  # Play the alarm sound
            return img, detected_text, "Invalid Number Plate"
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        st.stop()

# Streamlit app setup
st.title("Number Plate Recognition and Alarm System")

# File uploader for image
uploaded_file = st.file_uploader("Upload a number plate image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Process the uploaded image
    processed_image, detected_text, status = process_image(uploaded_file)

    # Display results
    if status == "Valid Number Plate":
        st.success(f"Number Plate Detected: {detected_text} (Valid)")
    elif status == "Invalid Number Plate":
        st.error(f"Number Plate Detected: {detected_text} (Invalid)")
        st.warning("Alarm Sound Triggered: Invalid Number Plate")
    else:
        st.error(status)

    # Display the processed image with annotations
    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
