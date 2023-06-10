from flask import Flask, jsonify, request
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
from mtcnn import MTCNN
import os
import random

app = Flask(__name__)

model = load_model('model.h5')

def crop_faces(image):
    nparr = np.fromstring(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Membuat objek detektor MTCNN
    detector = MTCNN()

    # Deteksi wajah
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        return []

     # Crop gambar menggunakan deteksi wajah
    cropped_images = []
    for face in faces:
        x, y, width, height = face['box']
        cropped_image = img[y:y+height, x:x+width]
        cropped_images.append(cropped_image)

    return cropped_images


def predict_face_shapes(images):
    predicted_shapes = []
    for image in images:
        will_process = Image.fromarray(image).resize((150, 150))
        img_array = np.array(will_process)
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        label_names = ['Square', 'Oblong', 'Round', 'Oval', 'Heart']
        predicted_label = label_names[np.argmax(predictions)]
        predicted_shapes.append(predicted_label)

    return predicted_shapes

def check_face_and_predict(dirty_image):
    cropped_images = crop_faces(dirty_image)
    if len(cropped_images) == 0:
        return "Wajah tidak terdeteksi"
    elif len(cropped_images) > 1:
        return "Hanya menerima satu inputan wajah"

    predicted_shapes = predict_face_shapes(cropped_images)
    return predicted_shapes


# Fungsi untuk merekomendasikan potongan rambut berdasarkan bentuk wajah
def recommend_hairstyles(face_shape):
    hairstyles_dir = "C:\\Users\\ayuad\\Documents\\ADIANI\MSIB Bangkit 2023\\api\\recommendation" # Path to the directory containing hairstyle models

    subdirectories = {
        "Oblong": ["Bergelombang", "Bob", "Layer", "Poni Samping"],
        "heart": ["Bob", "Poni Samping", "Updo", "Texturizing"],
        "Round": ["Bob Panjang", "Layer", "Poni Samping"],
        "Oval": ["Lurus", "Pixie", "Poni"],
        "Square": ["Half Knot", "Side Parted", "Side Parted Chignon"]
    }

    recommended_hairstyles = []

    if face_shape in subdirectories:
        subdirectory_names = subdirectories[face_shape]
        for subdirectory_name in subdirectory_names:
            subdirectory_path = os.path.join(hairstyles_dir, face_shape, subdirectory_name)

            # Get a list of hairstyle model files in the subdirectory
            hairstyle_files = os.listdir(subdirectory_path)

            # Randomly select 5 hairstyle models from the list
            recommended_hairstyle_files = random.sample(hairstyle_files, k=5)

            # Add the recommended hairstyles to the list
            for hairstyle_file in recommended_hairstyle_files:
                recommended_hairstyles.append(os.path.join(subdirectory_path, hairstyle_file))

    return recommended_hairstyles

@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return jsonify({
            "error": "No file uploaded",
        })

    file = request.files['files']
    if file.filename == '':
        return jsonify({
            "error": "No file selected",
        })

    # Memeriksa wajah dan melakukan prediksi jika terdeteksi satu wajah
    face_result = check_face_and_predict(file.read())
    if isinstance(face_result, list):
        face_shape = face_result[0]
        recommended_hairstyles = recommend_hairstyles(face_shape)
        random_recommendations = random.sample(recommended_hairstyles, 5)
        response = {
            "prediction": face_shape,
            "hairstyles": random_recommendations[:5]
        }
    else:
        response = {
            "error": face_result,
        }

    return jsonify(response)


@app.route('/')
def hello():
    return "Server Already Running"

if __name__ == "__main__":
    app.run(debug=True)

#code face detect before
    #  Membuat objek detektor MTCNN
    # detector = MTCNN()
    # # Deteksi wajah
    # faces = detector.detect_faces(img)
    # if len(faces) == 0:
    #     return "Wajah tidak terdeteksi"
    # elif len(faces) > 1:
    #     return "Hanya menerima satu inputan wajah"
    # return 1

    # haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # faces = haar.detectMultiScale(
    # gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#code before
    # def predict(dirty_image):
    #     img = Image.open(dirty_image)

    #     will_process = img.resize((150, 150))
    #     img_array = np.array(will_process)
    #     img_array = np.expand_dims(img_array, axis=0)

    #     predictions = model.predict(img_array)
    #     label_names = ['Square', 'Oblong', 'Round', 'Oval', 'Heart']
    #     predicted_label = label_names[np.argmax(predictions)]

    #     return predicted_label

#tidak terpakai
    # nparr = np.frombuffer(dirty_image, np.uint8)
    # img = Image.open(BytesIO(nparr))
    # haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # faces = haar.detectMultiScale(
    #     img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # for (x, y, w, h) in faces:
    #     x -= 15
    #     y -= 50
    #     w += 50
    #     h += 75

    #     x = max(0, x)
    #     y = max(0, y)
    #     w = min(img.shape[1] - x, w)
    #     h = min(img.shape[0] - y, h)

    #     cropped_img = img[y:y+h, x:x+w, :]
    # will_process = Image.fromarray(img)

    # def recommendation(label):
    #     return[
    #         "image 1",
    #         "image 2",
    #         "image 3",
    #         "image 4",
    #         "image 5",
    #     ]

#route before
    # @app.route('/predict', methods=['GET', 'POST'])
    # def index():
    #     if request.method == 'POST':
    #         file = request.files['files']
    #         face = check_face(file.read())
    #         if face == 1:
    #             value = predict(file)
    #             return jsonify({
    #                 "prediction": value,
    #             })
    #         return jsonify({
    #             "result": face,
    #         })
