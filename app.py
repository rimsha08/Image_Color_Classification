import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image 

# Load trained SVM model
with open('color_classification.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Set Streamlit page configuration
st.set_page_config(page_title="Color Classifier", layout="wide")

# Sidebar with compact layout
with st.sidebar:
    st.markdown("<h4 style='margin-bottom: 10px;'>📌 About</h4>", unsafe_allow_html=True)
    with st.expander("ℹ️ How it works", expanded=True):
        st.markdown(
            "Upload an image, and a trained SVM model will predict the dominant color from these 8 categories."
        )

    st.markdown("---")
    with st.expander("🛠 Tech Used", expanded=False):
        st.markdown("✅ Streamlit  \n✅ OpenCV  \n✅ Scikit-learn")

    st.markdown("---")
    with st.expander("🎨 Color Categories", expanded=True):
        color_categories = {
            "Orange": "#FFA500",
            "Violet": "#8A2BE2",
            "Red": "#FF0000",
            "Blue": "#0000FF",
            "Green": "#008000",
            "Black": "#000000",
            "Brown": "#A52A2A",
            "White": "#FFFFFF"
        }

        for color, hex_code in color_categories.items():
            st.markdown(
                f"<div style='display: flex; align-items: center; margin-bottom: 5px;'>"
                f"<div style='width: 14px; height: 14px; background: {hex_code}; border-radius: 3px; margin-right: 8px;'></div>"
                f"<span style='font-size: 14px;'>{color}</span></div>",
                unsafe_allow_html=True
            )

# Main title and layout
st.title("🎨 Image Color Classification")
st.write("Upload an image, and the model will predict the dominant color.")

# Function to preprocess the image (must match training preprocessing)
def preprocess_image(image):
    img_array = np.array(image)  # Convert PIL image to NumPy array
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR (for OpenCV compatibility)
    img_array = cv2.resize(img_array, (64, 64))  # Resize to match training size
    img_array = img_array / 255.0  # Normalize
    return img_array.flatten()  # Flatten to match training features

# Upload image
uploaded_image = st.file_uploader("📤 Upload an image for prediction", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Create two columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)
    
    with col2:
        # Preprocess image
        features = preprocess_image(image)
        
        # Make prediction
        prediction = svm_model.predict([features])
        
        # Define category labels (ensure these match your training labels)
        categories = ['Orange', 'Violet', 'Red', 'Blue', 'Green', 'Black', 'Brown', 'White']
        
        # Get predicted color
        predicted_label = categories[prediction[0]]
        
        # Display prediction result with styled message
        st.markdown(
            f"""
            <div style="padding: 20px; background-color: {color_categories.get(predicted_label, 'white')}; border-radius: 4px; text-align: center; 
            font-size: 22px; font-weight: bold; color: {'black' if predicted_label in ['White', 'Yellow'] else 'white'}; 
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
                🎯 Predicted Color: {predicted_label}
            </div>
            """,
            unsafe_allow_html=True
        )



