from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import model_from_json

st.title("Breast Medical Image Analysis")
class_labels = [
'Malignant',
'Normal/Benign'
]

models=["Xception","Model from scratch"]
model_choice = st.selectbox("Select Model",models)

def load_model(model_name):
    f = Path(f"breastMNIST/{model_name}.json")
    model_structure = f.read_text()
    model = model_from_json(model_structure)
    model.load_weights(f"breastMNIST/{model_name}.h5")
    return model

def upload_predict(upload_image,model_name):
    #processing image to fit neural network
    img = ImageOps.fit(upload_image, (28,28))
    img_array = np.asarray(img)
    img_array = img_array/255.0
    img_array = np.expand_dims(img_array, axis=-1)
    images = np.expand_dims(img_array, axis=0)
    results = load_model(model_name).predict(images)
    single_result = results[0]
    #prediction
    test_predict_labels = np.where(single_result > 0.5, 1, 0)
    class_label = class_labels[test_predict_labels[0]]
    # return class_label
    st.write("The image is classified as", class_label)
    # st.write("The similarity score is approximately", results[class_label])

if model_choice=='Xception':
    with st.form("my form",clear_on_submit=True):
        file = st.file_uploader("Upload ultrasound image of breast to be classified using Xception. \U0001F447", type=["jpg", "png"])
        submitted = st.form_submit_button("UPLOAD!")
    if submitted and file is not None:
        st.write("UPLOADED!")
        with st.spinner('Model is being loaded..'):
            model = load_model('xception')
        image = Image.open(file)
        st.image(image)
        predictions = upload_predict(image, 'xception')

elif model_choice=='Model from scratch':
    with st.form("my form",clear_on_submit=True):
        file = st.file_uploader("Upload ultrasound image of breast to be classified using model designed from scratch. \U0001F447", type=["jpg", "png"])
        submitted = st.form_submit_button("UPLOAD!")
    if submitted and file is not None:
        st.write("UPLOADED!")
        with st.spinner('Model is being loaded..'):
            model = load_model('model_from_scratch_breast')
        image = Image.open(file)
        st.image(image)
        predictions = upload_predict(image, 'model_from_scratch_breast')

