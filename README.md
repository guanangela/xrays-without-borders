# Xrays Without Borders
UC Berkeley MIDS Capstone Project, Summer 2022

## Problem Statement
We built a machine learning model which takes in a chest x-ray and predicts whether an abnormality is present. Our goal is to assist radiologists in detecting cardiomegaly early in an interpretable way.

Xrays Without Borders strives to assist physicians in efficiently diagnosing cardiomegaly, train radiologists, and provide interpretability to patients. For traveling clinics and global medical trips in crisis settings, this application offers an in-hospital experience.

## Model
DenseNet-121 model trained on ImageNet with a fully connected sigmoid in the final layer. In addition, we extracted the last convolutional layer and inspect the activation before the features are mapped to classification logits to apply GradCam, which visualizes the intermediate features by producing a coarse localization map that highlights important regions in the image for predicting the cardiomegaly or no finding.

## Launching the Website
```
# Here is the code for the web UI code. In addition, we run the machine learning inference.
# Once you have a model, we can serve your model with our inference pipeline locally instead of sending 
# another request to a different server.
cd xrays-without-borders

# Install dependencies
pip install -r requirements.txt

# Run the web app
flask run 
```

Then go to http://127.0.0.1:5000/ in your browser.

## Files
  * `app.py`: Flask App in Python
  * `inference.py`: Contains helper functions for running machine learning inference
  * `requirements.txt`: Packages and dependencies required
  * `weights_model.h5`: Trained Model
  * `Procfile`: File for Flask web app
  * `Aptfile`: File for opencv-headless
  * `templates`: Folder containing all html files for website

