import os
import numpy as np
import cv2
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = load_model('model/age_gender_race_model.h5')

# Class labels
gender_labels = ['Male', 'Female']
race_labels = ['White', 'Black', 'Asian', 'Indian', 'Other']

# Prediction function
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (96, 96))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    age_pred, gender_pred, race_pred = model.predict(img)

    age = round(float(age_pred[0]), 1)
    gender = gender_labels[int(gender_pred[0] > 0.5)]
    race = race_labels[np.argmax(race_pred[0])]
    return age, gender, race

# Route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['photo']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            age, gender, race = predict_image(filepath)

            return render_template('index.html', image_path=filepath,
                                   age=age, gender=gender, race=race)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
