import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import base64
import cv2
import tempfile

# Cargar el modelo
model = load_model('modelo.keras')

# Lista de nombres de clases en el mismo orden que el modelo las recibió
class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def circle_crop(img, sigmaX=30):
    height, width, depth = img.shape
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img

def load_ben_color(uploaded_file, img_size = 512, sigmaX=10):
    # Crear un archivo temporal y guardar el contenido del archivo cargado en él
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()

    # Leer la imagen
    image = cv2.imread(temp_file.name)

    # Convertir la imagen a espacio de color RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Recortar la imagen
    image = crop_image_from_gray(image)

    # Redimensionar la imagen
    image = cv2.resize(image, (img_size, img_size))

    # Aplicar un filtro Gaussiano a la imagen redimensionada
    image=cv2.addWeighted( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

    # Devolver la imagen recortada y redimensionada
    return image

st.title("!DETECCION DE RETINOPATIA DIABETICA!")

uploaded_file = st.file_uploader("Cargar Imagen", type=["jpg", "png"])

if uploaded_file is not None:
    # Convertir la imagen a un array de numpy
    img = Image.open(uploaded_file)
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Realizar la predicción
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]

    # Recortar y redimensionar la imagen
    image = load_ben_color(uploaded_file)
    cv2.imwrite('image.jpg', image)

    # Mostrar la imagen procesada
    st.image('image.jpg', caption='Imagen Procesada', use_column_width=True)

    # Mostrar el mensaje de respuesta
    st.write(f"Resultado: {predicted_class_name}")

