from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import pygame

app = Flask(__name__)

model = load_model('hammad_recognition.h5')

celebrity_dict = {0: 'Not_hammad', 1: 'hammad'}

model.make_predict_function()

def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    pygame.time.wait(2000)  # Adjust this delay as needed

def predict_label(img_path):
    img = load_img(img_path, color_mode="grayscale", target_size=(128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    pred = model.predict(img)
    pred = np.argmax(pred)
    pred_celebrity = celebrity_dict[pred]

    if pred_celebrity == 'hammad':
        play_audio("output.mp3")
    elif pred_celebrity == 'Not_hammad':
        play_audio("output1.mp3")

    return pred_celebrity

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
    return render_template("index.html", celebrity=p, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
