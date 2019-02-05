#https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Time%20Series%20Forecastings.ipynb
#https://github.com/rjgiedt/Cardiovascular_Death_Prediction/blob/master/SimpleNN_Baseline.ipynb
#https://ccforum.biomedcentral.com/articles/10.1186/cc11396
#https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
import warnings
#import itertools
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import statsmodels.api as sm
#import matplotlib
#from pylab import rcParams
from statsmodels.tsa.ar_model import AR

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
#matplotlib.rcParams['axes.labelsize'] = 14
#matplotlib.rcParams['xtick.labelsize'] = 12
#matplotlib.rcParams['ytick.labelsize'] = 12
#matplotlib.rcParams['text.color'] = 'k'
#rcParams['figure.figsize'] = 18, 8

df = pd.read_csv("CHARTEVENTS_HR_FILTERED.csv")
#,SUBJECT_ID,HADM_ID,ICUSTAY_ID,CHARTTIME,HEART_RATE
heart_rate_36 = df.loc[df['SUBJECT_ID'] == 36]
heart_rate_36 = heart_rate_36[['CHARTTIME','HEART_RATE']]

#Make the index a time datatype, make only one reading per hour and fill in missing values
heart_rate_36['CHARTTIME'] = pd.to_datetime(heart_rate_36['CHARTTIME'])
heart_rate_36 = heart_rate_36.set_index('CHARTTIME')
heart_rate_36_resampled = heart_rate_36.resample('H').mean()
heart_rate_36_resampled = heart_rate_36_resampled.interpolate(method='linear')

print ("Original data points: " + str(len(heart_rate_36)))
print ("Resampled hourly data points: " + str(len(heart_rate_36_resampled)))
print (plt.style.available)

#Autoregression (AR)
model = AR(heart_rate_36_resampled)
model_fit = model.fit()
heart_rate_36_forecast = model_fit.predict(len(heart_rate_36_resampled), len(heart_rate_36_resampled)+24)

plt.figure(figsize=(16,8))
plt.plot(heart_rate_36, label='Original')
plt.plot(heart_rate_36_resampled, label='Resampled')
plt.plot(heart_rate_36_forecast, label='AR Forecast')
plt.legend(loc='best')
plt.show()