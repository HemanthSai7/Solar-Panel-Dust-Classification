import streamlit as st
import time
import numpy as np
from ModelScript import ModelScript

LABEL_DICT={
    0:'Clean Solar Panel',
    1:'Dirty Solar Panel'
}


st.set_page_config(layout="wide")
st.title("Solar Panel Soiling Detection")
st.text('This Web App is used to classify Clean vs Dusty solar module.')
st.text("Choose Your Option: 1)Img_Upload 2)Img_URL")
st.text("Choose Your Model : 1)DenseNet   2)AlexNet")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("./assets/solar.jpg")
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 


method=st.sidebar.selectbox('Capture or Upload an Image',('Upload Image','Capture Image'))
if method=='Upload Image':
    image_file=st.file_uploader('Upload an Image',type=['jpg','png','jpeg'])
else:
    image_file=st.camera_input("Capture an Image")

if image_file:
    progress_bar=st.progress(0)
    for i in range(100):
        time.sleep(0.001)
        progress_bar.progress(i+1)
    st.info('Image uploaded successfully')
    st.image(image_file.getvalue())

if image_file is not None:
    ModelScript=ModelScript()
    model=ModelScript.load_model()
    with st.spinner("Please wait..."):
        image=  ModelScript.preprocessing(image_file.getvalue())
        prediction=ModelScript.predict(image,model)
        st.balloons()
    st.success(LABEL_DICT[prediction])
else:
    st.sidebar.warning("Please upload or capture an image")