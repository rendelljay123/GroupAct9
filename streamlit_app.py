import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

css = """
body {
    background-image: url('./assets/leaf-bg.jpg');
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-image: linear-gradient(rgba(0, 153, 51, 0.2), rgba(255, 255, 255, 0.2));
    background-size: cover;
    background-position: center;
}
"""

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = np.asarray(image)
    image = image / 255.0
    image_reshape = np.reshape(image, (1, 256, 256, 3))
    prediction = model.predict(image_reshape)
    return prediction

model_paths = {
    'tomato': './model/Tomato_Model.h5',
    'cotton': './model/Cotton_Model.h5',
    'potato': './model/Potato_Model.h5'
}

default_model_path = './model/Tomato_Model.h5'
model = load_model(default_model_path)

class_names = {
    'tomato': {0: 'Tomato_Early_blight', 1: 'Tomato_Leaf_Mold', 2: 'Tomato_healthy'},
    'cotton': {0: 'diseased cotton leaf', 1: 'diseased cotton plant', 2: 'fresh cotton leaf', 3: 'fresh cotton plant'},
    'potato': {0: 'Potato___Early_blight', 1: 'Potato___Late_blight', 2: 'Potato___healthy'}
}

disease_info = {
    'tomato': {
        'Tomato_Early_blight': "Early blight is a common fungal disease that affects tomato plants. It is caused by the fungus Alternaria solani.",
        'Tomato_Leaf_Mold': "Tomato leaf mold is a foliar disease that primarily affects the leaves...",
        'Tomato_healthy': "Your tomato plant looks healthy!",
    },
    'cotton': {
        'diseased cotton leaf': "Diseased cotton leaves often show symptoms like yellowing and wilting...",
        'diseased cotton plant': "Cotton plant diseases can lead to reduced yield and fiber quality...",
        'fresh cotton leaf': "Your cotton leaf appears healthy! In a thriving state, a fresh cotton leaf showcases vibrant green color, smooth texture, and well-defined veins. This is a positive sign of a leaf that is actively contributing to the plant's photosynthesis and overall well-being.",
        'fresh cotton plant': "Your cotton plant appears healthy! A healthy cotton plant exhibits robust growth with lush, green foliage. It stands tall with sturdy stems, indicating a well-nourished and disease-free condition. Proper care has contributed to the plant's vitality and potential for optimal cotton production.",
        'other cotton disease 1': "Description for other cotton disease 1...",
        'other cotton disease 2': "Description for other cotton disease 2...",
    },
    'potato': {
        'Potato___Early_blight': "Early blight in potatoes can cause dark lesions on leaves. It is primarily caused by the fungus Alternaria solani. Proper spacing, fungicides, and crop rotation can help manage early blight.",
        'Potato___Late_blight': "Late blight is a serious disease of potato and tomato crops, caused by the oomycete pathogen Phytophthora infestans. Early detection, proper plant spacing, and fungicide applications are key preventive measures.",
        'Potato___healthy': "Your potato plant looks healthy! Healthy potato plants typically have vibrant green leaves, well-spaced foliage, and show no signs of discoloration or lesions. Regular monitoring and good agricultural practices contribute to maintaining plant health.",
    }
}

st.write("# Plant Disease Detection")
st.write("### CPE 028 - DevOps")
st.write("Baltazar, Rendell Jay; Dalmacio, Andre Christian; Makiramdam, Releigh; Orbeta, John Mark; Co, Jericho")

plant_type = st.selectbox("Select Plant Type", ('tomato', 'cotton', 'potato'))

file = st.file_uploader("Upload an image", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    try:
        model_path = model_paths.get(plant_type)
        if model_path:
            model = load_model(model_path)
        else:
            st.text("Invalid plant type")

        image = Image.open(file) if file else None
        if image:
            st.image(image, use_column_width=True)
            prediction = import_and_predict(image, model)
            class_labels = class_names.get(plant_type)
            if class_labels:
                class_index = np.argmax(prediction)
                class_name = class_labels.get(class_index)
                if class_name:
                    string = f"The {plant_type} plant is {class_name}"
                    st.success(string)

                    with st.expander("Learn More About the Disease"):
                        disease_description = disease_info.get(plant_type, {}).get(class_name, "No information available.")
                        st.write(disease_description)
                else:
                    st.text("Invalid class index")
            else:
                st.text("Invalid class labels for the selected plant type")
        else:
            st.text("Invalid file. Please upload a valid image file.")
    except Exception as e:
        st.text("Error occurred while processing the image.")
        st.text(str(e))
