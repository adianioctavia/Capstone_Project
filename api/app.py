from flask import Flask, jsonify, request
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

app = Flask(__name__)

model = load_model('model.h5')

def check_face(image):
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = haar.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return "Wajah tidak terdeteksi"
    return 1

def predict(dirty_image):
    nparr = np.frombuffer(dirty_image, np.uint8)
    img = Image.open(BytesIO(nparr))

    haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = haar.detectMultiScale(
        img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        x -= 15
        y -= 50
        w += 50
        h += 75

        x = max(0, x)
        y = max(0, y)
        w = min(img.shape[1] - x, w)
        h = min(img.shape[0] - y, h)

        cropped_img = img[y:y+h, x:x+w, :]

    will_process = Image.fromarray(cropped_img)
    will_process = will_process.resize((150, 150))
    img_array = np.array(will_process)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    label_names = ['Square', 'Oblong', 'Round', 'Oval', 'Heart']
    predicted_label = label_names[np.argmax(predictions)]

    return predicted_label

@app.route('/predict', methods=['POST'])
def index():
    if request.method == 'POST':
        file = request.files['files']
        face = check_face(file.read())
        value = predict(file.read())
        if face == 1:
            return jsonify({
                "prediction": value,
            })
        return jsonify({
            "result": face,
        })

@app.route('/')
def hello():
    return "Server Already Running"

if __name__ == "__main__":
    app.run(debug=True)
