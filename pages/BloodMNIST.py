from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from keras.models import model_from_json

st.title("Blood Medical Image Analysis")
class_labels = [
 'basophil',
 'eosinophil',
 'erythroblast',
 'immature granulocytes(myelocytes, metamyelocytes and promyelocytes)',
 'lymphocyte',
 'monocyte',
 'neutrophil',
 'platelet'
]

models=["VGG16","Model from scratch"]
model_choice = st.selectbox("Select a Model",models)

def load_model(model_name):
    f = Path(f"bloodMNIST/{model_name}.json")
    model_structure = f.read_text()
    model = model_from_json(model_structure)
    model.load_weights(f"bloodMNIST/{model_name}.h5")
    return model

def upload_predict(upload_image,model_name):
    if(model_name=='vgg16'):
        img = ImageOps.fit(upload_image, (32, 32))
    else:
        img = ImageOps.fit(upload_image, (28, 28))
    img_array = np.asarray(img)
    images = np.expand_dims(img_array, axis=0)
    results = load_model(model_name).predict(images)
    single_result = results[0]
    most_likely_class_index = int(np.argmax(single_result))
    class_likelihood = single_result[most_likely_class_index]
    class_label = class_labels[most_likely_class_index]
    # return class_label
    st.write("The image is classified as", class_label)
    st.write("The similarity score is approximately", class_likelihood)


if model_choice=='VGG16':
    with st.form("my form",clear_on_submit=True):
        file = st.file_uploader("Upload an image of blood cells to be classified using VGG16. \U0001F447", type=["jpg", "png"])
        submitted = st.form_submit_button("UPLOAD!")
    if submitted and file is not None:
        st.write("UPLOADED!")
        with st.spinner('Model is being loaded..'):
            model = load_model('vgg16')
        image = Image.open(file)
        st.image(image)
        predictions = upload_predict(image, 'vgg16')


elif model_choice=='Model from scratch':
    with st.form("my form",clear_on_submit=True):
        file = st.file_uploader("Upload an image of blood cells to be classified using model designed from scratch. \U0001F447", type=["jpg", "png"])
        submitted = st.form_submit_button("UPLOAD!")
    if submitted and file is not None:
        st.write("UPLOADED!")
        with st.spinner('Model is being loaded..'):
            model = load_model('model_from_scratch_blood')
        image = Image.open(file)
        st.image(image)
        predictions = upload_predict(image, 'model_from_scratch_blood')



