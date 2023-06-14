from google.cloud import storage
from flask import Flask, jsonify, request
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
from mtcnn import MTCNN
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
    storage_client = storage.Client.from_service_account_json("keys.json")
    blobs = storage_client.list_blobs("haircut_recommendation")

    recommended_hairstyles = []
    for blob in blobs:
        url = blob.public_url
        _, label, _, _ = blob.name.rsplit('/', 3)

        if (label == face_shape):
            recommended_hairstyles.append(url)

    recommended_hairstyles = random.sample(recommended_hairstyles, k=5)

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
