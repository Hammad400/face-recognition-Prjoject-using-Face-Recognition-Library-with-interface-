from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import subprocess

app = Flask(__name__)

model = load_model('hammad_recognition.h5')
mp3_file = "urdu_output.mp3"
mp3_file1 = "urdu_output1.mp3"

face_cascade=cv2.CascadeClassifier(r'opencv_haarcascades\haarcascade_frontalface_default.xml')
eye_cascade =cv2.CascadeClassifier(r'opencv_haarcascades\haarcascade_eye.xml')

celebrity_dict = {0: 'Not_hammad', 1: 'hammad'}
model.make_predict_function()

def predict_label(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    img = cv2.resize(img, (128, 128))
    img = img_to_array(img)
    img = img.reshape(1, 128, 128, 1)
    img = img / 255.0
    pred = model.predict(img)
    pred = np.argmax(pred.round(2))
    pred_celebrity = celebrity_dict[pred]

    if pred_celebrity == 'hammad':
        subprocess.Popen(["start", "cmd", "/c", mp3_file], shell=True)

    if pred_celebrity == 'Not_hammad':
        subprocess.Popen(["start", "cmd", "/c", mp3_file1], shell=True)

    return pred_celebrity

def get_cropped_image_if_2_eyes(img):
    # img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        return roi_color

# Initialize the camera
camera = cv2.VideoCapture(0)

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index2.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        ret, frame = camera.read()  # Read a frame from the camera
        c_image=get_cropped_image_if_2_eyes(frame)
        img_path = "static/live_image.jpg"
        cv2.imwrite(img_path, c_image)  # Save the frame as an image
        p = predict_label(c_image)

    return render_template("index2.html", celebrity=p, img_path=img_path)
    
@app.route("/close_camera", methods=['POST'])
def close_camera():
    global camera  # Assuming camera is a global variable
    if camera:
        camera.release()  # Release the camera resource
    return "Camera closed successfully"



if __name__ == '__main__':
    app.run(debug=True)
cv2.waitKey(0)
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
