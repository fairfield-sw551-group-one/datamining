from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import csv
from flask_cors import CORS

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify, json
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
CORS(app)

# Model saved with Keras model.save()
MODEL_PATH = 'models/your_model.h5'

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

from keras.applications.resnet50 import ResNet50
print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/timeforcasting/predict', methods=['POST'])
def timeforcasingPredict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files["file"]
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        basepath, 'uploads/timeforcasting', secure_filename(f.filename))
        f.save(file_path)



        #process input data through NN model
        
        
        # return output as JSON  
        #response = jsonify(data)
        #return response

        #mock response
        timestamp = ['1:00','2:00','3:00','4:00','5:00','6:00','7:00']
        bpm = ['78','92','84','82','90','87','94']
        response = jsonify(timestamp=timestamp,bpm=bpm)
        return response
    
    return None


@app.route('/ecgclassification/predict', methods=['POST'])
def ecgClassPredict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files["file"]
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        basepath, 'uploads/ecgclassification', secure_filename(f.filename))
        f.save(file_path)



        #process input data through NN model
        
        
        # return output as JSON  
        #response = jsonify(data)
        #return response
        
        
        #mock response
        with open('mockecg.json') as mockecg:
            data = json.load(mockecg)
            return jsonify(data)
    
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()