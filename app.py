from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import subprocess

app = Flask(__name__)



model = load_model('hammad_recognition.h5')
mp3_file = "urdu_output.mp3"
mp3_file1 = "urdu_output1.mp3"

celebrity_dict = {0:'Not_hammad', 1:'hammad'}
model.make_predict_function()

def predict_label(img_path):
	img = load_img(img_path, grayscale=True)
	img = img.resize((128, 128))
	img = np.array(img)
	img=img.reshape(128,128,1)
	img=img/255.0
	pred = model.predict(img.reshape(1, 128, 128, 1))
	pred = np.argmax(pred.round(2))
	pred_celebrity=celebrity_dict[pred]

	if pred_celebrity == 'hammad':

		subprocess.Popen(["start", "cmd", "/c", mp3_file], shell=True)

	if pred_celebrity == 'Not_hammad':

		subprocess.Popen(["start", "cmd", "/c", mp3_file1], shell=True)


	return pred_celebrity


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")



@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", celebrity = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)