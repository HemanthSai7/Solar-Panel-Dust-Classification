import json
import time
import numpy as np

import requests
import streamlit as st
from streamlit_lottie import st_lottie

from ModelScript import ModelScript

LABEL_DICT={
    0:'Clean Solar Panel',
    1:'Dirty Solar Panel'
}


st.set_page_config(layout="wide")
st.title("Solar Panel Soiling Detection")

st.info('Machine learning approach to detect dust on solar panels in UAE a contribution toward optimizing cleaning plan.')

def load_lottiefile(filepath:str):
    with open(filepath,'r') as f:
        return json.load(f)

lottie_solar_code=load_lottiefile('src/assets/solar.json')

col1,col2,col3=st.columns([1,4.5,0.2])
with col2:
    st_lottie(
    lottie_solar_code,
    speed=1,
    reverse=True,
    loop=True,
    quality='high',
    height=500,
    width=500,
    key='Knowledge Bank'
)
    

st.sidebar.subheader('This Web App is used to classify Clean vs Dusty solar module.')


model_name=st.sidebar.selectbox("Select the model",('EfficientNet','DenseNet','AlexNet'),help="More models will be added soon")
method=st.sidebar.selectbox('Capture or Upload an Image',('Upload Image','Capture Image'))

if method=='Upload Image':
    image_file=st.sidebar.file_uploader('Upload an Image',type=['jpg','png','jpeg'])
else:
    image_file=st.sidebar.camera_input("Capture an Image")

if image_file:
    progress_bar=st.sidebar.progress(0)
    for i in range(100):
        time.sleep(0.001)
        progress_bar.progress(i+1)
    st.sidebar.info('Image uploaded successfully')
    st.image(image_file.getvalue())

if image_file is not None:
    ModelScript=ModelScript(model_name=model_name)
    model=ModelScript.load_model()
    with st.spinner("Please wait..."):
        image=  ModelScript.preprocessing(image_file.getvalue())
        prediction=ModelScript.predict(image,model)
        st.balloons()
    st.success(LABEL_DICT[prediction])
else:
    st.sidebar.warning("Please upload or capture an image")


