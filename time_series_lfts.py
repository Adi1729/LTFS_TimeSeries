'''
https://www.youtube.com/watch?v=e8Yw4alG16Q
'''
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import datetime
rcParams['figure.figsize']=10,6
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
import os   

os.chdir(r'/home/aditya/Analytics Vidhya Practice/LTFS/')

dataset = pd.read_csv('train_fwYjLYX.csv')
dataset['Year'] = dataset['application_date'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d').year)
dataset['Month'] = dataset['application_date'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d').month)
dataset['Day'] = dataset['application_date'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d').day)
dataset['application_date'] = dataset['application_date'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d'))

ADF = pd.DataFrame(columns = range(7))
state_list ='TAMIL NADU'
segment =2

state_list_2=['ASSAM', 'BIHAR', 'CHHATTISGARH', 'GUJARAT', 'HARYANA',
       'JHARKHAND', 'KERALA', 'KARNATAKA', 'MAHARASHTRA',
       'MADHYA PRADESH', 'ORISSA', 'PUNJAB', 'TAMIL NADU', 'TRIPURA',
       'UTTAR PRADESH', 'WEST BENGAL']

state_list_2_1=[ 'CHHATTISGARH', 'GUJARAT',
       'KERALA', 'KARNATAKA', 'MAHARASHTRA',
       'MADHYA PRADESH', 'ORISSA', 'PUNJAB', 'TAMIL NADU', 
       'UTTAR PRADESH', 'WEST BENGAL']

for state_name in state_list_2_1:
        
    state = dataset[(dataset['segment']==segment)][['application_date','case_count']].reset_index()
    state = state.groupby(['application_date'])['case_count'].sum().reset_index()
    indexedDataset = state.set_index(['application_date']) 
    dftest = adfuller(indexedDataset['case_count'],autolag ='AIC')
    dfoutput = pd.Series(dftest[0:4],index = ['Test Statistics','p-value','#Lags Used','Number of Observation Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    ADF =ADF.append(pd.DataFrame(data = {state_name : dfoutput.values}).T)

ADF.columns = dfoutput.index

ADF.to_csv('./Output/ADF_segment_2.csv')

## plot graph

plt.xlabel("Date")
plt.ylabel("Case Count")
plt.plot(np.log(indexedDataset+1))

test_Seasonality(indexedDataset)
plot_estimate_trend(indexedDataset)
plot_acf_pac(indexedDataset)

plt.xlabel("Date")
plt.ylabel("Case Count")
plt.plot(np.log(indexedDataset+1))

test_Seasonality(np.log(indexedDataset+1))
plot_estimate_trend(np.log(indexedDataset+1))
plot_acf_pac(np.log(indexedDataset+1))

rolmean =np.log(indexedDataset+1).rolling(window = 365).mean()
datasetLogScaleMinusMovingAverage = np.log(indexedDataset+1) -rolmean

test_Seasonality(datasetLogScaleMinusMovingAverage)
plot_estimate_trend(datasetLogScaleMinusMovingAverage)
plot_acf_pac(datasetLogScaleMinusMovingAverage)

# remove NaN values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)

# ExponentialDecayWeightedAverage
exponentialDecayWeightedAverage = np.log(indexedDataset+1).ewm(halflife = 365,min_periods=0,adjust= True).mean()
plt.plot(np.log(indexedDataset+1))
plt.plot(exponentialDecayWeightedAverage)


# AR Model
from statsmodels.tsa.arima_model import ARIMA
indexedDataset = np.log(indexedDataset+1)
model = ARIMA(indexedDataset, order = (0,0,0))
results_AR = model.fit(disp = -1)
plt.plot(indexedDataset, color = 'green')
plt.plot(results_AR.fittedvalues,color ='red')

a= results_AR.fittedvalues.values 
b= indexedDataset.values
a= np.exp(a)-1
b= np.exp(b)- 1
plt.title('MAPE %.4f'%mean_absolute_percentage_error(a,b))
print('Plotting AR Model')

predictions_ARIMA = pd.Series(results_AR.fittedvalues, copy = True)
print(predictions_ARIMA.head())

# Convert to Cumulative Sum
predictions_ARIMA_cumsum = predictions_ARIMA.cumsum()

prediction_ARIMA_1 = pd.Series(indexedDataset['case_count'].iloc[0], index = indexedDataset.index)
prediction_ARIMA_1 = prediction_ARIMA_1.add(predictions_ARIMA_cumsum,fill_value=0)

start_index = datetime(2019,6,7)
end_index = datetime(2019,9,30)
fts <- forecast(model, level = c(90))
valid =90
forecast = np.exp(results_AR.forecast(steps=87)[1])-1
forecast_segment1 = forecast

# invert the differenced forecast to something usable
forecast = inverse_difference(X, forecast, days_in_year)
print('Forecast: %f' % forecast)

import statsmodels.api as sm
mod = sm.tsa.statespace.SARIMAX(indexedDataset,
                                order=(0, 0, 1),
                                seasonal_order=(1, 1, 1, 30),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(18, 8))
plt.show()

pred = results.get_prediction(start=pd.to_datetime('2019-06-07'), dynamic=False)
pred_ci = pred.conf_int()
ax =indexedDataset.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Retail_sold')
plt.legend()
plt.show()

y_forecasted = pred.predicted_mean
y_truth = indexedDataset
mse = ((y_forecasted - y_truth) ** 2).mean()
mean_absolute_percentage_error(y_forecasted,y_truth)
print('The Mean Squared Error is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))


from pyramid.arima import auto_arima
stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())