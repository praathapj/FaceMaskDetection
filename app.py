import streamlit as st
import cv2
import numpy as np
from camera_input_live import camera_input_live
import tensorflow as tf
from tensorflow import keras


frame = camera_input_live()

st.image(frame)

#st.write(frame)
# load model

# source: https://analyticsindiamag.com/deploy-your-deep-learning-based-image-classification-model-with-streamlit/
# load model

@st.cache_resource()
def load_model():

    DLmodel=tf.keras.models.load_model('faceMaskDetect_model_lite.hdf5')
    return DLmodel

model = load_model()

if frame is not None:
    # To read image file buffer with OpenCV:
    bytes_data = frame.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    img = cv2.resize(cv2_img,(224,224))
    img = img/255 # standardize img
    pred = (model.predict(img.reshape(1,224,224,3)) > 0.5).astype("int32")

    if pred[0][0] == 0:
        st.success("Mask ON")
    else:
        st.error("No Mask")

st.write(" ")
st.subheader("Click for more information")
# Source: https://stackoverflow.com/questions/74003574/how-to-create-a-button-with-hyperlink-in-streamlit
url = 'https://github.com/praathapj/FaceMaskDetection'

st.markdown(f'''
<a href={url}><button style="background-color:GreenYellow;">GitHub</button></a>
''',unsafe_allow_html=True)
