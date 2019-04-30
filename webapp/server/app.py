from __future__ import division, print_function

from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template, jsonify, json
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import cv2
import biosppy
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
import keras.backend as K
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import csv
from flask_cors import CORS

# Time Series
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Needed for ecg model

# Keras

# Flask utils

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
    print("timeforcast/print call debug enabled")
    if request.method == 'POST':
        # Get the file from post request
        f = request.files["file"]
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads/timeforcasting', secure_filename(f.filename))
        f.save(file_path)

        # Load file and perform basic validation
        if not os.path.isfile(file_path):
            print("400: File not found")
            return jsonify(message="File not found"), 400
        if os.path.getsize(file_path) == 0:
            print("400: Empty file found")
            return jsonify(message="Empty file found"), 400

        df = pd.read_csv(file_path)
        if 'HEART_RATE' not in df.columns:
            print("400: File missing HEART_RATE")
            return jsonify(message="File missing HEART_RATE"), 400
        if 'CHART_TIME' not in df.columns:
            print("400: File missing CHART_TIME")
            return jsonify(message="File missing CHART_TIME"), 400

        heart_rate = df[['CHART_TIME', 'HEART_RATE']]

        # Make the index a time datatype, make only one reading per hour and fill in missing values
        heart_rate['CHART_TIME'] = pd.to_datetime(heart_rate['CHART_TIME'])
        heart_rate = heart_rate.set_index('CHART_TIME')
        heart_rate_resampled = heart_rate.resample('H').mean()
        heart_rate_resampled = heart_rate_resampled.interpolate(
            method='linear')

        # Use last 24 hours as test data
        raw_values = heart_rate_resampled.values
        train, test = raw_values[0:-24], raw_values[-24:]
        sc = MinMaxScaler()
        train_sc = sc.fit_transform(train)
        test_sc = sc.transform(test)
        train_sc_df = pd.DataFrame(train_sc, columns=['Y'])
        test_sc_df = pd.DataFrame(test_sc, columns=['Y'])
        for s in range(1, 2):
            train_sc_df['X_{}'.format(s)] = train_sc_df['Y'].shift(s)
            test_sc_df['X_{}'.format(s)] = test_sc_df['Y'].shift(s)
        X_train = train_sc_df.dropna().drop('Y', axis=1)
        y_train = train_sc_df.dropna().drop('X_1', axis=1)
        X_test = test_sc_df.dropna().drop('Y', axis=1)
        y_test = test_sc_df.dropna().drop('X_1', axis=1)
        X_train = X_train.as_matrix()
        y_train = y_train.as_matrix()
        X_test = X_test.as_matrix()
        y_test = y_test.as_matrix()
        regressor = SVR(kernel='rbf')
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        #r2_test = mean_squared_error(y_test, y_pred)
        K.clear_session()

        # Specify two layer nueral network
        model = Sequential()
        model.add(Dense(50, input_shape=(
            X_test.shape[1],), activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(50, input_shape=(X_test.shape[1],), activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=12, epochs=24, verbose=0)
        y_pred = model.predict(X_test)

        # Get data for response
        timestamp = heart_rate_resampled.index
        inputBPM = heart_rate['HEART_RATE'].tolist()
        predictBPM = sc.inverse_transform(y_pred).tolist()

        maxBPM = max(inputBPM)
        minBPM = min(inputBPM)

        # Round up/down to nearest 5
        minBPM = minBPM - (minBPM % 5)
        maxBPM = maxBPM + (maxBPM % 5)

        # Pad list with nul values
        predictBPM = [None] * (len(inputBPM)-24) + predictBPM

        # TODO generate actual response
        # mock response
        #timestamp = ['1:00','2:00','3:00','4:00','5:00','6:00','7:00']
        #inputBPM = [78, 92, 84, 82, None, None, None]
        #predictBPM = [None, None, None, None, 92, 90, 86]
        #maxBPM = 92
        #minBPM = 78

        response = jsonify(timestamp=timestamp.strftime('%m/%d/%Y %H:%M').tolist(),
                           inputBPM=inputBPM, predictBPM=predictBPM, min=minBPM, max=maxBPM)

        print("timeforcast/print returning calculated values")
        return response

    print("timeforcast/print returning nothing")
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

        # Load file and perform basic validation
        if not os.path.isfile(file_path):
            print("400: File not found")
            return jsonify(message="File not found"), 400
        if os.path.getsize(file_path) == 0:
            print("400: Empty file found")
            return jsonify(message="Empty file found"), 400

        df = pd.read_csv(file_path)
        if 'ECG_VALUE' not in df.columns:
            print("400: File missing ECG_VALUE")
            return jsonify(message="File missing ECG_VALUE"), 400

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
    csv_data = csv['ECG_VALUE']
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
        beat['start'] = str(indices[count][0])
        beat['end'] = str(indices[count][1])
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
