#References:
#https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Time%20Series%20Forecastings.ipynb
#https://github.com/rjgiedt/Cardiovascular_Death_Prediction/blob/master/SimpleNN_Baseline.ipynb
#https://ccforum.biomedcentral.com/articles/10.1186/cc11396
#https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
#https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
#https://github.com/neelabhpant/Deep-Learning-in-Python

import warnings
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
import keras
import keras.backend as K
from keras.layers import Dense
from keras.models import Sequential
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot

warnings.filterwarnings("ignore")
#print (plt.style.available)
plt.style.use('fivethirtyeight')

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", help="Specify a model to forecast.")
args = parser.parse_args()

###############################################
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

def adj_r2_score(r2, n, k):
	return 1-((1-r2)*((n-1)/(n-k-1)))
###############################################

df = pd.read_csv("CHARTEVENTS_HR_FILTERED.csv")
#,SUBJECT_ID,HADM_ID,ICUSTAY_ID,CHARTTIME,HEART_RATE
heart_rate_36 = df.loc[df['SUBJECT_ID'] == 36]
heart_rate_36 = heart_rate_36[['CHARTTIME','HEART_RATE']]

#Make the index a time datatype, make only one reading per hour and fill in missing values
heart_rate_36['CHARTTIME'] = pd.to_datetime(heart_rate_36['CHARTTIME'])
heart_rate_36 = heart_rate_36.set_index('CHARTTIME')
heart_rate_36_resampled = heart_rate_36.resample('H').mean()
heart_rate_36_resampled = heart_rate_36_resampled.interpolate(method='linear')

#print ("Original data points: " + str(len(heart_rate_36)))
#print ("Resampled hourly data points: " + str(len(heart_rate_36_resampled)))

model_type = args.model
plot = True

if model_type.upper() == 'AR':
	#Autoregression (AR)
	model = AR(heart_rate_36_resampled)
	model_fit = model.fit()
	heart_rate_36_forecast = model_fit.predict(len(heart_rate_36_resampled), len(heart_rate_36_resampled)+24)
elif model_type.upper() == 'MA':
	#Moving Average (MA)
	model = ARMA(heart_rate_36_resampled, order=(0, 1))
	model_fit = model.fit(disp=False)
	heart_rate_36_forecast = model_fit.predict(len(heart_rate_36_resampled), len(heart_rate_36_resampled)+24)
elif model_type.upper() == 'ARMA':
	#Autoregressive Moving Average (ARMA)
	model = ARMA(heart_rate_36_resampled, order=(2, 1))
	model_fit = model.fit(disp=False)
	heart_rate_36_forecast = model_fit.predict(len(heart_rate_36_resampled), len(heart_rate_36_resampled)+24)
elif model_type.upper() == 'ARIMA':
	#Autoregressive Integrated Moving Average (ARIMA)
	model = ARIMA(heart_rate_36_resampled, order=(1, 1, 1))
	model_fit = model.fit(disp=False)
	heart_rate_36_forecast = model_fit.predict(len(heart_rate_36_resampled), len(heart_rate_36_resampled)+24, typ='levels')
elif model_type.upper() == 'SARIMA':
	#Seasonal Autoregressive Integrated Moving-Average (SARIMA)
	model = SARIMAX(heart_rate_36_resampled, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
	model_fit = model.fit(disp=False)
	heart_rate_36_forecast = model_fit.predict(len(heart_rate_36_resampled), len(heart_rate_36_resampled)+24)
elif model_type.upper() == 'SES':
	#Simple Exponential Smoothing (SES)
	model = SimpleExpSmoothing(heart_rate_36_resampled)
	model_fit = model.fit()
	heart_rate_36_forecast = model_fit.predict(len(heart_rate_36_resampled), len(heart_rate_36_resampled)+24)
elif model_type.upper() == 'HWES':
	#Holt Winter’s Exponential Smoothing (HWES)
	model = ExponentialSmoothing(heart_rate_36_resampled)
	model_fit = model.fit()
	heart_rate_36_forecast = model_fit.predict(len(heart_rate_36_resampled), len(heart_rate_36_resampled)+24)
elif model_type.upper() == 'LSTM':
	#Long Short Term Memory Nueral Network
	raw_values = heart_rate_36_resampled.values
	diff_values = difference(raw_values, 1)
	supervised = timeseries_to_supervised(diff_values, 1)
	supervised_values = supervised.values
	
	train, test = supervised_values[0:-24], supervised_values[-24:]
	scaler, train_scaled, test_scaled = scale(train, test)
	lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
	train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
	lstm_model.predict(train_reshaped, batch_size=1)
	
	#Walk Forward Validation
	predictions = list()
	for i in range(len(test_scaled)):
		#1 step forecast
		X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		yhat = invert_scale(scaler, X, yhat)
		yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
		predictions.append(yhat)
		expected = raw_values[len(train) + i + 1]
		print('Hour=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
	
	#Performance
	rmse = sqrt(mean_squared_error(raw_values[-24:], predictions))
	print('Test RMSE: %.3f' % rmse)
	plt.figure(figsize=(16,8))
	plt.plot(raw_values[-24:], label='Resampled')
	plt.plot(predictions, label='Forecast')
	plt.legend(loc='best')
	plt.show()
	plot = False
elif model_type.upper() == 'RK':
	#Regresision Using Keras NN
	raw_values = heart_rate_36_resampled.values
	train, test = raw_values[0:-24], raw_values[-24:]
	sc = MinMaxScaler()
	train_sc = sc.fit_transform(train)
	test_sc = sc.transform(test)
	train_sc_df = pd.DataFrame(train_sc, columns=['Y']) #index=train.index
	test_sc_df = pd.DataFrame(test_sc, columns=['Y']) #index=test.index
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
	print('Train size: (%d x %d)'%(X_train.shape[0], X_train.shape[1]))
	print('Test size: (%d x %d)'%(X_test.shape[0], X_test.shape[1]))
	regressor = SVR(kernel='rbf')
	regressor.fit(X_train, y_train)
	y_pred = regressor.predict(X_test)
	r2_test = mean_squared_error(y_test, y_pred)
	K.clear_session()
	model = Sequential()
	model.add(Dense(50, input_shape=(X_test.shape[1],), activation='relu', kernel_initializer='lecun_uniform'))
	model.add(Dense(50, input_shape=(X_test.shape[1],), activation='relu'))
	model.add(Dense(1))
	model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')
	model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=1)
	y_pred = model.predict(X_test)
	print('R-Squared: %f'%(mean_squared_error(y_test, y_pred)))
	plt.figure(figsize=(16,8))
	plt.plot(y_test, label='Resampled')
	plt.plot(y_pred, label='Forecast')
	plt.legend(loc='best')
	plt.show()
	plot = False
else:
	sys.exit("Error: Invalid model '" + model_type + "' specified!")

if plot:
	plt.figure(figsize=(16,8))
	plt.plot(heart_rate_36, label='Original')
	plt.plot(heart_rate_36_resampled, label='Resampled')
	plt.plot(heart_rate_36_forecast, label=model_type + ' Forecast')
	plt.legend(loc='best')
	plt.show()
