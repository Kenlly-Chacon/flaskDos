from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import base64
from flask_cors import CORS
import cv2
from image_processing import load_ben_color


app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "https://retinopatia-diabetica.netlify.app"}})

# Cargar el modelo
model = load_model('modelo.keras')

# Lista de nombres de clases en el mismo orden que el modelo las recibió
class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        file = request.files['image']
        filepath = 'image.jpg'
        file.save(filepath)

        print('¡Imagen recibida y guardada con éxito!')

        file = request.files['image']
        # Convertir la imagen a un array de numpy
        img = Image.open(file.stream)
        img = img.resize((256, 256))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Realizar la predicción
        prediction = model.predict(img)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        print(predicted_class_name)

        # Recortar y redimensionar la imagen
        image = load_ben_color(filepath)
        cv2.imwrite(filepath, image)
        # time.sleep(7)

        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        data = {
            "image": encoded_string,
            'message': predicted_class_name
        }

        return jsonify(data)
    else:
        print('No se recibió ninguna imagen')
        return 'No se encontró ninguna imagen en la petición'


if __name__ == '__main__':
    app.run()

