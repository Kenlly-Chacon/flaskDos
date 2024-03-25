import cv2
import numpy as np

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img

# Función para recortar una imagen en forma de círculo
def circle_crop(img, sigmaX=30):

    # Leer la imagen
    img = cv2.imread(img)

    # Convertir la imagen a espacio de color RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Recortar la imagen
    img = crop_image_from_gray(img)

    # Calcular el centro de la imagen y el radio del círculo
    height, width, depth = img.shape
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))

    # Crear una imagen en blanco del mismo tamaño que la imagen de entrada
    circle_img = np.zeros((height, width), np.uint8)

    # Dibujar un círculo en la imagen en blanco
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)

    # Usar la imagen del círculo como una máscara para recortar la imagen de entrada
    img = cv2.bitwise_and(img, img, mask=circle_img)

    # Recortar la imagen de nuevo
    img = crop_image_from_gray(img)

    # Aplicar un filtro Gaussiano a la imagen recortada
    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)

    # Devolver la imagen recortada
    return img

# Función para cargar una imagen en color
def load_ben_color(path, img_size = 512, sigmaX=10):

    # Leer la imagen
    image = cv2.imread(path)

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