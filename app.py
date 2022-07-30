import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from inference import *
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static/images')
app.config['UPLOAD_FOLDER'] = os.path.join('static')
app.config['SECRET_KEY'] = "key"
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT']= False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

ALLOWED_EXTENSIONS = {'tif', 'tiff', 'jpg', 'jpeg', 'png', 'gif'}
labels = ['Cardiomegaly', 'No Finding']

# Checks whether a file is valid and can be uploaded.
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
# Import our model 
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
def my_custom_func():
    def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
        def weighted_loss(y_true, y_pred):
            # initialize loss to zero
            loss = 0.0
            for i in range(len(pos_weights)):
                # print(tf.cast(y_pred, tf.float32))
                y_true_var = tf.cast(y_true[:, i], tf.float32)
                y_pred_var = tf.cast(y_pred[:, i], tf.float32)
                loss_pos = -1 * K.mean(pos_weights[i] * y_true_var * K.log(y_pred_var + epsilon))
                loss_neg = -1 * K.mean(neg_weights[i] * (1 - y_true_var) * K.log(1 - y_pred_var + epsilon))
                loss += loss_pos + loss_neg
            return loss
        return weighted_loss
get_custom_objects().update({'my_custom_func': my_custom_func})

model = tf.keras.models.load_model('AJ_2_model.h5')

# helper functions: image processing 
def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img

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


# Main method that handles a user upload request.
# When an image is sent it:
# 1. Ensures the image is a valid image
# 2. Saves the image in the 'images/' folder
# 3. Runs inference on the saved image
#    NOTE: During inference, we create a new image with the bounding boxes
#    In your case, we could create a new template and return both the image
#    and the classification. 
# 4. Returns the result of inference
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    class_id, class_name, saved_filename = None, None, None
    if request.method == 'POST':
        print(request.files)
        # 1. Ensure request is well-formed.
        if 'file' not in request.files:
            print('No image present.')
            return redirect(request.url)
        
        file = request.files['file']

        # 1. Ensure the image exists.
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            print('No selected file.')
            return redirect(request.url)

        # If we're here, the user uploaded the correct file.
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # 2. Save the image to a path in the server's directory.
            # NOTE: This is not the safest thing to do.
            saved_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'images/' + filename)
            # print(saved_filename)
            file.save(saved_filename)
            # print(filename)

            # 3. Run inference.
            with open(saved_filename, 'rb') as f:
                image_bytes = f.read()
            # img_bytes = file.read()
            # img = prepare_image(image_bytes)
            # return jsonify(prediction=predict_result(img))
            # prediction = predict_result(img)
            # print('prediction->>>>>>>>>>>>>>>>>>>>>>>', prediction)

            # for i in range(len(labels)):
                # grad_cam_path = 'static/images/grad_cam.png' 
            grad_cam_path_1 = 'static/images/grad_cam_1.png'
            grad_cam_path_2 = 'static/images/grad_cam_2.png'
            pred_prob_1 = compute_gradcam(model, saved_filename, 'Cardiomegaly', grad_cam_path_1, layer_name='bn')
            # pred_prob_2 = compute_gradcam(model, saved_filename, 'No Finding', grad_cam_path_2, layer_name='bn')
            if pred_prob_1 > 0.5:
                predicted_label = 'Cardiomegaly'
                return render_template("result.html", prediction_val = predicted_label, saved_filename = 'images/' + filename, gradcam_heatmap_1 = 'images/grad_cam_1.png') #, gradcam_heatmap_2 = 'images/grad_cam_2.png')    

            else:
                predicted_label = 'No Finding'
                compute_gradcam(model, saved_filename, 'No Finding', grad_cam_path_2, layer_name='bn')
                return render_template("result.html", prediction_val = predicted_label, saved_filename = 'images/' + filename, gradcam_heatmap_1 = 'images/grad_cam_2.png') #, gradcam_heatmap_2 = 'images/grad_cam_2.png')    
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)