# streamlit_audio_recorder y whisper by Alfredo Diaz - version Mayo 2024

# En VsC seleccione la version de Python (recomiendo 3.9)
#CTRL SHIFT P  para crear el enviroment (Escriba Python Create Enviroment) y luego venv

#o puede usar el siguiente comando en el shell
#Vaya a "view" en el menú y luego a terminal y lance un terminal.
#python -m venv env

#Verifique que el terminal inicio con el enviroment o en la carpeta del proyecto active el env.
#cd D:\gondola\env\Scripts\
#.\activate

#Debe quedar asi: (.venv) D:\proyectos_ia\gondola>

#Puedes verificar que no tenga ningun libreria preinstalada con
#pip freeze
#Actualicie pip con pip install --upgrade pip

#pip install tensorflow==2.15 La que tiene instalada Google Colab o con la versión qu fué entrenado el modelo
#Verifique se se instaló numpy, no trate de instalar numpy con pip install numpy, que puede instalar una version diferente
#pip install streamlit
#Verifique se se instaló no trante de instalar con pip install pillow
#Esta instalacion se hace si la requiere pip install opencv-python

# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st
import tensorflow as tf # TensorFlow is required for Keras to work
from PIL import Image
import numpy as np
#import cv2

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Reconocimiento de gondola",
    page_icon = "icon.png",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) # Oculta el código CSS de la pantalla, ya que están incrustados en el texto de rebajas. Además, permita que Streamlit se procese de forma insegura como HTML

#st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache_resource
def load_model():
    model=tf.keras.models.load_model('./gondola_model.h5')
    return model
with st.spinner('Modelo está cargando..'):
    model=load_model()



with st.sidebar:
        st.image('logo.jpg')
        st.title("Universidad Autónoma de Bucaramanga")
        st.subheader("Este programa esta hecho para reconocer productos de gondola")

st.image('pgondola.jpeg')
st.title("Reconocimiento de imagen")
st.write("Reconocimiento de imagen para productos de gondola/supermercado")
st.write("""
         # Detección de gondola
         """
         )


def import_and_predict(image_data, model, class_names):

    image_data = image_data.resize((180, 180))

    image = tf.keras.utils.img_to_array(image_data)
    image = tf.expand_dims(image, 0) # Create a batch


    # Predecir con el modelo
    prediction = model.predict(image)
    index = np.argmax(prediction)
    score = tf.nn.softmax(prediction[0])
    class_name = class_names[index]

    return class_name, score


class_names = open("./clases.txt", "r").readlines()

img_file_buffer = st.camera_input("Capture una foto para identificar un producto")
if img_file_buffer is None:
    st.text("Por favor tome una foto")
else:
    image = Image.open(img_file_buffer)
    st.image(image, use_column_width=True)

    # Realizar la predicción
    class_name, score = import_and_predict(image, model, class_names)

    # Mostrar el resultado

    if np.max(score)>0.5:
        st.subheader(f"Tipo de Producto: {class_name}")
        st.text(f"Puntuación de confianza: {100 * np.max(score):.2f}%")
    else:
        st.text(f"No se pudo determinar el tipo de producto")