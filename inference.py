"""
Handles inference using a model.
See https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
"""
import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, flash, request, render_template, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import cv2

# imagenet_class_index = json.load(open('imagenet_class_index.json'))

# Make sure to pass `pretrained` as `True` to use the pretrained weights:
# model = models.resnet18(pretrained=True)
# model = tf.keras.models.load_model('weights_model.h5')

# Since we are using our model only for inference, switch to `eval` mode:
# model.eval()

def prepare_image(img):
  img = Image.open(io.BytesIO(img))
  img = img.resize((224, 224))
  img = np.array(img)
  img = np.expand_dims(img, 0)
  return img

# def predict_result(img):
#   pred = model.predict(img)[0]
#   print(pred)
#   if pred[0] > 0.5:
#     return 'Cardiomegaly' 
#   else:
#     return 'No Finding'

# def get_prediction(filename):
# 	with open(filename, 'rb') as f:
# 		image_bytes = f.read()
# 		tensor = transform_image(image_bytes=image_bytes)
# 	outputs = model.forward(tensor)
# 	# The tensor y_hat will contain the index of the predicted class id.
# 	# However, we need a human readable class name. For that we need a class 
# 	# id to name mapping.
# 	_, y_hat = outputs.max(1)
# 	predicted_idx = str(y_hat.item())
# 	return imagenet_class_index[predicted_idx]

# 	return "Unsure", -1

# GradCam
def load_image_2(image_dir):
# load image via tf.io
  image_generator = tf.io.read_file(image_dir)  

# convert to tensor (specify 3 channels explicitly since png files contains additional alpha channel)
# set the dtypes to align with pytorch for comparison since it will use uint8 by default
  tensor = tf.io.decode_image(image_generator, channels=3, dtype=tf.dtypes.float32)
# (384, 470, 3)

# resize tensor to 320 x 320
  tensor = tf.image.resize(tensor, [320, 320])
# (320, 320, 3)

# add another dimension at the front to get NHWC shape
  img_tensor = tf.expand_dims(tensor, axis=0)
# (1, 320, 320, 3)
  #print(img_tensor)
  return img_tensor


def grad_cam(input_model, image, layer_name, H=320, W=320):
  """GradCAM method for visualizing input saliency."""
  '''y_c = input_model.output[0, cls]
  conv_output = input_model.get_layer(layer_name).output
  grads = K.gradients(y_c, conv_output)[0] ### gradients error? 

  gradient_function = K.function([input_model.input], [conv_output, grads])

  output, grads_val = gradient_function([image])
  output, grads_val = output[0, :], grads_val[0, :, :, :]
  
  weights = np.mean(grads_val, axis=(0, 1))
  cam = np.dot(output, weights)'''

  ### AJ EDITS ###
  
  #model = load_model(os.path.join(model_folder, "custom_model.h5"))
  #image = image.load_img(image_path) 
  #img_tensor = image.img_to_array(image)
  #print(image) #print(type(image))
  img_tensor = image
  #print(img_tensor)
  
  preds = input_model.predict(img_tensor)
  model_prediction = input_model.output[:, np.argmax(preds[0])]
  
  conv_layer = input_model.get_layer(layer_name) 
  heatmap_model = tf.keras.models.Model([input_model.inputs], [conv_layer.output, input_model.output])

  with tf.GradientTape() as tape:
    conv_output, predictions = heatmap_model(img_tensor) ### have to edit... img_tensor
    loss = predictions[:, np.argmax(predictions[0])]
  
  grads = tape.gradient(loss, conv_output)

  castConvOutputs = tf.cast(conv_output > 0, "float32")
  castGrads = tf.cast(grads > 0, "float32")
  guidedGrads = castConvOutputs * castGrads * grads
  # My goal was to create the class activation map for single image, so we are skipping axis=0 which is meant to have the batch_size of images at axis=0
  convOutputs = conv_output[0]
  guidedGrads = guidedGrads[0]

  weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
  cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

  # Process CAM
  '''cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)'''
  cam = cv2.resize(cam.numpy(), (W, H))
  cam = np.maximum(cam, 0)
  cam = cam / cam.max()
  print(type(cam))
  print(cam)
  return cam


def compute_gradcam(model, image_dir, label, saved_path, layer_name='bn'):
        pre_processed_input_2 = load_image_2(image_dir)
        predictions = model.predict(pre_processed_input_2)
        with open(image_dir, 'rb') as f:
          image_bytes = f.read()
        # for i in range(len(labels)):
        # heatmap for Cardiomegaly 
        if label == 'Cardiomegaly':
          label_index = 0
        else:
          label_index = 1
        print(f"Generating gradcam for class {label}") 
        gradcam = grad_cam(model, pre_processed_input_2, layer_name) # changed preprocessed_input -> pre_processed_input_2
        plt.title(f"{label}: p={predictions[0][label_index]:.3f}")
        plt.axis('off')
        # plot original image + heatmap
        original_image = tf.keras.preprocessing.image.load_img(image_dir, target_size=(320, 320))
        plt.imshow(original_image, cmap='gray')
        plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][label_index]))
        plt.savefig(saved_path)
        # returns predicted label confidence/probability
        return predictions[0][label_index]

        

     