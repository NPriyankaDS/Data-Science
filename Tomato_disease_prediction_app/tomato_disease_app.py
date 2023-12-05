import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from keras.models import load_model
import numpy as np
import base64
from io import BytesIO
from PIL import Image

st.title("Tomato disease prediction")
st.markdown("This app is for predicting the type of a tomato disease in a plant.")

image_file = st.file_uploader("Drop the picture of the plant leaf",type = ['jpg','jpeg','jfif'])

class_names = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

  

def model_load():
    model = load_model('tomatos.h5')
    return model

def prediction(model,image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    #st.write(img_array)
    img_array /= 255.0
    #st.write(img_array)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


model = model_load()

if image_file is not None:
    image_base64 = base64.b64encode(image_file.read())
    
    im_bytes = base64.b64decode(image_base64)   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
    img = Image.open(im_file)   # img is now PIL Image object
    IMAGE_SIZE =(256,256)
    image = img.resize(IMAGE_SIZE)
    #st.write(image)
    #image /= 255.0
    st.image(image)
    button = st.button("Predict")
    
    if button:
        predicted_class, confidence = prediction(model,image)
        st.success(f"The disease predicted is {predicted_class} with confidence of {confidence}")


