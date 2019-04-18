from __future__ import division, print_function
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
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify, json, abort
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

DEBUG = 1

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
    print_debug ("timeforcast/print call debug enabled")
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
        if 'CHARTTIME' not in df.columns:
            print("400: File missing CHARTTIME")
            return jsonify(message="File missing CHARTTIME"), 400

        heart_rate = df[['CHARTTIME','HEART_RATE']]

        # Make the index a time datatype, make only one reading per hour and fill in missing values
        heart_rate['CHARTTIME'] = pd.to_datetime(heart_rate['CHARTTIME'])
        heart_rate = heart_rate.set_index('CHARTTIME')
        heart_rate_resampled = heart_rate.resample('H').mean()
        heart_rate_resampled = heart_rate_resampled.interpolate(method='linear')

        # Use last 24 hours as test data
        raw_values = heart_rate_resampled.values
        train, test = raw_values[0:-24], raw_values[-24:]
        sc = MinMaxScaler()
        train_sc = sc.fit_transform(train)
        test_sc = sc.transform(test)
        train_sc_df = pd.DataFrame(train_sc, columns=['Y'])
        test_sc_df = pd.DataFrame(test_sc, columns=['Y'])
        for s in range(1,2):
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
        model.add(Dense(50, input_shape=(X_test.shape[1],), activation='relu', kernel_initializer='lecun_uniform'))
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

        #TODO generate actual response
        #mock response
        #timestamp = ['1:00','2:00','3:00','4:00','5:00','6:00','7:00']
        #inputBPM = [78, 92, 84, 82, None, None, None]
        #predictBPM = [None, None, None, None, 92, 90, 86]
        #maxBPM = 92
        #minBPM = 78

        response = jsonify(timestamp=timestamp.strftime('%m/%d/%Y %H:%M').tolist(), inputBPM=inputBPM, predictBPM=predictBPM, min=minBPM, max=maxBPM)

        print_debug ("timeforcast/print returning calculated values")
        return response

    print_debug ("timeforcast/print returning nothing")
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

def print_debug(message):
    if DEBUG:
        print(message)

if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()