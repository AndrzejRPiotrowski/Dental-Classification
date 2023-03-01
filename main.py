import streamlit as st
from objectDetection import *
from detectron2.engine import DefaultPredictor
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from PIL import Image
import tqdm
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import keras
from tensorflow import keras
from tensorflow.python.keras.models import *
# from tensorflow.keras import preprocessing
from keras import preprocessing
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout,Embedding,LSTM,GRU
from keras.preprocessing.image import ImageDataGenerator
# from keras.layers.embeddings import Embedding
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

fig = plt.figure()
st.title('NeptuneDent - Sprawdz stan swoich zębów')
st.markdown("-------------------------------------------------------------------------------------------------------")
st.markdown("Model oparty na przetwarzaniu obrazu przy użyciu algorytmu AI (Predykcja) : Najczęstrze problemy pacjentów")
st.markdown("-------------------------------------------------------------------------------------------------------")
st.markdown("Model został oparty na danych własnych gabinetu Agnieszka Nowińska Indywidualna Praktyka Dentystyczna")
st.markdown("-------------------------------------------------------------------------------------------------------")

st.markdown("'Próchnica = CAVITY', 1: 'Martwy Ząb = COLD SORES' 2:'Brak uzębienia = DEAD TOOTH',3:' Parodontoza = GINGIVITY',4:' Zdrowy - HEALTHY'}")
st.markdown("-------------------------------------------------------------------------------------------------------")
st.markdown("Model rozpoznaje następujące schorzenia : Próchnica")
st.markdown("Jak działa rozpoznawanie")
st.markdown("-------------------------------------------------------------------------------------------------------")
st.markdown("Model rozpoznaje następujące schorzenia : Martwy Ząb")
st.markdown("Jak działa rozpoznawanie")
st.markdown("-------------------------------------------------------------------------------------------------------")
st.markdown("Model rozpoznaje następujące schorzenia : Brak uzębienia")
st.markdown("Jak działa rozpoznawanie")
st.markdown("-------------------------------------------------------------------------------------------------------")
st.markdown("Model rozpoznaje następujące schorzenia : Zdrowy ząb")
st.markdown("Jak działa rozpoznawanie")
st.markdown("-------------------------------------------------------------------------------------------------------")
st.markdown("Granty Norweskie 2023 - Norway Grants 2023")
#x = st.slider('x')  # 👈 this is a widget
#st.write(x, 'squared is', x * 
     
#img_file_buffer = st.camera_input("Zrób zdjęcie")
#if img_file_buffer is not None:
    # To read image file buffer as bytes:
   # bytes_data = img_file_buffer.getvalue()
    # Check the type of bytes_data:
    # Should output: <class 'bytes'>
  #  st.write(type(bytes_data))
         
#st.date_input(label, value=None, min_value=None, max_value=None, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible")
#st.camera_input(label, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible")

def func_1(x):
    detector = Detector(model_type=x)
    image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
    if image_file is not None:
        file_details = {"FileName":image_file.name,"FileType":image_file.type}
        st.write(file_details)
        img = Image.open(image_file)
        st.image(img, caption='Uploaded Image.')
        with open(image_file.name,mode = "wb") as f: 
            f.write(image_file.getbuffer())         
        st.success("Saved File")
        detector.onImage(image_file.name)
        img_ = Image.open("result.jpg")
        st.image(img_, caption='Proccesed Image.')

def main():
    with st.expander("About the App"):
        st.markdown( '<p style="font-size: 30px;"><strong>Welcome to my Object Detection App!</strong></p>', unsafe_allow_html= True)
        st.markdown('<p style = "font-size : 20px; color : white;">This app was built using Streamlit, Detectron2 and OpenCv to demonstrate <strong>Object Detection</strong> in both videos (pre-recorded) and images.</p>', unsafe_allow_html=True)
        


    option = st.selectbox(
     'What Type of File do you want to work with?',
     ('Images', 'Videos'))

    #st.write('You selected:', option)
    if option == "Images":
        st.title('Object Detection for Images')
        st.subheader("""
This takes in an image and outputs the image with bounding boxes created around the objects in the image.
""")
        func_1('objectDetection')
    else:
        st.title('Object Detection for Videos')
        st.subheader("""
This takes in a video and outputs the video with bounding boxes created around the objects in the video.
""")
        func_2('objectDetection')


if __name__ == '__main__':
		main()
          
          
def predict(image):
    classifier_model = 'newclass04.h5'
    model=keras.models.load_model(classifier_model)



    test_image = image.resize((244, 244))
    test_image = np.array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = {0: 'CAVITY', 1: 'COLD SORES',2:"DEAD TOOTH",3:"GINGIVITY",4:"HEALTHY"}
    predictions = model.predict(test_image)
    print(predictions)
    a=np.array(max(predictions))
    print(class_names[np.argmax(a)])
    result = f"Dokładność uzyskana na podstawie predykcji {class_names[np.argmax(a)]} with {(max(a)*100).__round__(2)} %. "

    return result

# ========================================================


def pred(image):
    from keras.models import load_model
    from PIL import Image, ImageOps
    import numpy as np

    # Load the model
    model = load_model('keras_model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    class_names = {0: 'CAVITY', 1: 'COLD SORES',2:"DEAD TOOTH",3:"GINGIVITY",4:"HEALTHY"}
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    a = np.array(max(prediction))
    print(class_names[np.argmax(a)])
    result = f"Dokładność uzyskana na podstawie predykcji {class_names[np.argmax(a)]} with {(max(a) * 100).__round__(2)} %. "



    return result


file_uploaded = st.file_uploader("WSKAŻ PLIK", type=["png", "jpg", "jpeg"])
class_btn = st.button("DOKONAJ ANALIZY")
if file_uploaded is not None:
    image = Image.open(file_uploaded)
    st.image(image, caption='WGRAJ PLIK', use_column_width=True)

if class_btn:
    if file_uploaded is None:
        st.write("BŁĘDNY STAN - PROGRAM SIĘ ZRESTARTUJE")
    else:
        with st.spinner('ANALIZA W TOKU....'):
            plt.imshow(image)
            plt.axis("off")
            predictions = predict(image)
            time.sleep(1)
            st.success('ANALIZA ZAKOŃCZONA SUKCESEM.')
            st.write(predictions)



