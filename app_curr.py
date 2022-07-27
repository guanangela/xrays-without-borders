#  reference: https://www.youtube.com/watch?v=2e4STDACVA8
# pip install virtualenv
# virtualenv venv
# source venv/bin/activate

import os
from flask import Flask, flash, request, render_template, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename

# NOTE: Inference can be done by importing a method that acts on the image. 
from inference_pytorch import get_prediction
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import Keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D, Lambda
from keras.models import Model
from keras import backend as K
from keras.models import load_model
import tensorflow as tf; print(tf.__version__)
import keras; print(keras.__version__)

ALLOWED_EXTENSIONS = {'tif', 'tiff', 'jpg', 'jpeg', 'png', 'gif'}

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/demo')
def demo():
    return render_template("demo.html")

@app.route('/team')
def team():
    return render_template("team.html")

if __name__ == '__main__':
    app.run(debug=True)


app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static')
app.config['SECRET_KEY'] = "key"
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT']= False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

# Checks whether a file is valid and can be uploaded.
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Main method that handles a user upload request.
# When an image is sent it:
# 1. Ensures the image is a valid image
# 2. Saves the image in the 'images/' folder
# 3. Runs inference on the saved image
#    NOTE: During inference, we create a new image with the bounding boxes
#    In your case, we could create a new template and return both the image
#    and the classification. 
# 4. Returns the result of inference
# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     class_id, class_name, saved_filename = None, None, None
#     if request.method == 'POST':
#         print(request.files)
#         # 1. Ensure request is well-formed.
#         if 'file' not in request.files:
#             print('No image present.')
#             return redirect(request.url)
        
#         file = request.files['file']

#         # 1. Ensure the image exists.
#         # If the user does not select a file, the browser submits an
#         # empty file without a filename.
#         if file.filename == '':
#             print('No selected file.')
#             return redirect(request.url)

#         # If we're here, the user uploaded the correct file.
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             # 2. Save the image to a path in the server's directory.
#             # NOTE: This is not the safest thing to do.
#             saved_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(saved_filename)
#             print(filename)

#             # 3. Run inference.
#             class_id, class_name = get_prediction(saved_filename)
#             class_name = class_name.replace('_', ' ')
        

#         # 4. Send the new image back to the user.
#         return render_template("result.html", class_id = class_id, class_name = class_name, saved_filename = filename)    
#     return render_template('index.html')



import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request

model = tf.keras.models.load_model('weights_model.h5')

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img

def predict_result(img):
    return 1 if model.predict(img)[0][0] > 0.5 else 0

@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')

    if not file:
        return

    img_bytes = file.read()
    img = prepare_image(img_bytes)
    return jsonify(prediction=predict_result(img))
    

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True)



# @app.route('/upload')
# def upload():
#     return stuff()


# from flask import Flask
# from flask_s3 import FlaskS3

# app = Flask(__name__)
# app.config['FLASKS3_BUCKET_NAME'] = 'mybucketname'
# s3 = FlaskS3(app)

# s3://ec2-jupyter-notebook-us-west-1-2c6d383426024f808fc5c5368c189839/mass_nf_merged (1).csv