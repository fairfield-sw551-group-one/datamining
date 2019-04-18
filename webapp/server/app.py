from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import csv
from flask_cors import CORS

# Needed for ecg model
import biosppy
import pandas as pd
import cv2
import matplotlib.pyplot as plt

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
# print('Model loaded. Check http://127.0.0.1:5000/')


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

        # process input data through NN model

        # return output as JSON
        #response = jsonify(data)
        # return response

        # mock response
        timestamp = ['1:00', '2:00', '3:00', '4:00', '5:00', '6:00', '7:00']
        bpm = [78, 92, 84, 82, 90, 87, 94]
        response = jsonify(timestamp=timestamp, bpm=bpm)
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
        print("sending this path to the model: " + file_path)
        output = runModel(file_path)

        print("done.")
        print(output)
        # process input data through NN model

        # return output as JSON
        #response = jsonify(data)
        # return response

        # mock response
        # with open('mockecg.json') as mockecg:
        #     data = json.load(mockecg)
        #     return jsonify(data)

        return jsonify(output)

    return None


def runModel(file_path):
    print('loading ecg model...')
    model = load_model('ecgModel.hdf5')
    model._make_predict_function()
    print('Model loaded. Start serving...')
    output = []
    print("sending this path to the prediction: " + file_path)
    output = model_predict(file_path, model)
    return output


def model_predict(file_path, model):
    flag = 1
    output = []

    #index1 = str(path).find('sig-2') + 6
    #index2 = -4
    #ts = int(str(path)[index1:index2])

    # Array for each prediction class
    APC, NORMAL, LBB, PVC, PAB, RBB, VEB = [], [], [], [], [], [], []
    output.append(str(file_path))
    result = {"APC": APC, "Normal": NORMAL, "LBB": LBB,
              "PAB": PAB, "PVC": PVC, "RBB": RBB, "VEB": VEB}

    indices = []

    kernel = np.ones((4, 4), np.uint8)

    csv = pd.read_csv(file_path)
    csv_data = csv[' Sample Value']
    data = np.array(csv_data)
    signals = []
    count = 1
    peaks = biosppy.signals.ecg.christov_segmenter(
        signal=data, sampling_rate=200)[0]

    # Output object
    json_output = []
    #json_output['source_file'] = str(file_path)

    for i in (peaks[1:-1]):

        diff1 = abs(peaks[count - 1] - i)
        diff2 = abs(peaks[count + 1] - i)
        x = peaks[count - 1] + diff1//2
        y = peaks[count + 1] - diff2//2
        signal = data[x:y]
        signals.append(signal)
        count += 1
        indices.append((x, y))

    for count, i in enumerate(signals):
        beat = {}
        beat['beatId'] = count
        fig = plt.figure(frameon=False)
        plt.plot(i)
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        filename = 'fig' + '.png'
        fig.savefig(filename)
        im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.erode(im_gray, kernel, iterations=1)
        im_gray = cv2.resize(im_gray, (128, 128),
                             interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(filename, im_gray)
        im_gray = cv2.imread(filename)
        pred = model.predict(im_gray.reshape((1, 128, 128, 3)))
        pred_class = pred.argmax(axis=-1)
        if pred_class == 0:
            APC.append(indices[count])
            beat['type'] = 'Atrial Premature Contraction'
        elif pred_class == 1:
            NORMAL.append(indices[count])
            beat['type'] = 'Normal'
        elif pred_class == 2:
            LBB.append(indices[count])
            beat['type'] = 'Left Bundle Branch Block'
        elif pred_class == 3:
            PAB.append(indices[count])
            beat['type'] = 'Paced Beat'
        elif pred_class == 4:
            PVC.append(indices[count])
            beat['type'] = 'Premature Ventricular Contraction'
        elif pred_class == 5:
            RBB.append(indices[count])
            beat['type'] = 'Right Bundle Branch Block'
        elif pred_class == 6:
            VEB.append(indices[count])
            beat['type'] = 'Ventricular Escape'
        beat['timestamp'] = str(indices[count][0])
        beat['confidence'] = str(indices[count][1])
        json_output.append(beat)

    result = sorted(result.items(), key=lambda y: len(y[1]))[::-1]
    output.append(result)

    print('writing result ' + str(flag) + 'to output array')
    # print(json_output)
    # json_filename = 'data.txt'
    # with open(json_filename, 'a+') as outfile:
    #     json.dump(data, outfile)
    flag += 1

    # print(json_output)
    # with open(json_filename, 'r') as file:
    #     filedata = file.read()
    # filedata = filedata.replace('}{', ',')
    # with open(json_filename, 'w') as file:
    #     file.write(filedata)
    os.remove('fig.png')
    return json_output


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
