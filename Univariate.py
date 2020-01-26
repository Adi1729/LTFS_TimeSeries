import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np
 
import tools

'''
source: 
    https://www.kaggle.com/kanncaa1/time-series-prediction-tutorial-with-eda
    https://www.kaggle.com/rgrajan/time-series-exploratory-data-analysis-forecast
    
credits : Brandon Rohrer | End to End Machine Learning | Time Series Analysis

'''
def get_data(train = None, segment = None):
    
    segment_1 = train[train['segment']==segment]
    segment_1_describe =segment_1.describe()
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_cols = segment_1.select_dtypes(include=numerics).columns.tolist()
    
    segment_1.isnull().sum()
    segment_1[num_cols].describe().T
    
    #state = segment_1[segment_1['state']=='MAHARASHTRA']
    state= segment_1
    state['Year'] = state['application_date'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d').year)
    state['Month'] = state['application_date'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d').month)
    state['Day'] = state['application_date'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d').day)
    
         
    return state
    
def plot_trend(statelist = None, state = None):

    df= state
    stateseries = pd.DataFrame(df[(df['state'].\
        isin(statelist))][['application_date','state','case_count']].\
        dropna().\
        groupby(['application_date', 'state'])['state','case_count'].mean().unstack())
    stateseries.plot(figsize=(15,8), linewidth=3)
    plt.show()
    
     
#    
#    #Seaspnality
#    
#    sns.set()
#    season = df
#    season['Date'] = df['application_date'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d').day)
#    season['Year'] = df['application_date'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d').year)
#    season['Month'] =df['application_date'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d').month)
#    spivot = pd.pivot_table(season, index='Month', columns = 'Year', values = 'case_count', aggfunc=np.mean)
#    spivot.plot(figsize=(20,10), linewidth=3)
#    plt.show()
#    
#    # Doesnot give clear trend
#    
#    
#    days = season.groupby(['Month','Date'])['case_count'].mean().dropna()
#    days.plot(figsize=(10,8))
#    plt.show()
#    
#    # Autocorrelation
#    case = segment_1.groupby(['application_date'])['case_count'].mean().to_numpy()
#    plt.plot(tools.autocorrelation(case= case))
#
#    #No Correlation with prior day

def build_case_calendar(case = None, window = None):

    """
    Parameters:
    ----------------
    case : array of float
        Array of value(here, case) for which median is to be analysed
    window : int
        range of value in which median has to be observed
            
    Returns:
    --------------
    median_case_calendar : array of float
    
    """
        
    median_case_calendar  = np.zeros(366)
    ten_day_medians = np.zeros(case.size)
    lower_limit  = int(window/2)
    upper_limit = int((window-1)/2)
    for i_day in range(0,365):
        low_day = i_day - lower_limit
        high_day = i_day + upper_limit
        if low_day < 0:
            low_day += 365
        if high_day > 365:
            high_day += -365
        if low_day < high_day:
            i_window_days = np.where(
                np.logical_and(day_of_year >=  low_day,
                               day_of_year <= high_day))
        else:
            i_window_days = np.where(
                np.logical_or(day_of_year >=  low_day,
                               day_of_year <= high_day))
            
        ten_day_median = np.median(case[i_window_days])
        median_case_calendar[i_day] = ten_day_median
        ten_day_medians[np.where(day_of_year==i_day)]=ten_day_median
        
        if i_day == 364:
            ten_day_medians[np.where(day_of_year==365)] = ten_day_median
            median_case_calendar[365] = ten_day_median
#        print(i_day, low_day, high_day, i_window_days[0].size)

    return median_case_calendar, ten_day_medians


def predict(day, month, year, case_calendar):
    """
    For a for a given day, month, year predict the case count
    
    Parameters:
    --------------
    year, month, day : int
        The date of interest
    case_calendar : array of floats
        The case count for each day of year
        
    Returns:
    --------------
    prediction: float
    """
    
    doy= find_day_of_year(year,month,day)
    prediction = case_calendar[doy]
    
    return prediction

if __name__ =='__main__':
    os.chdir(r'/home/aditya/Analytics Vidhya Practice/LTFS/')

    train = pd.read_csv('train_fwYjLYX.csv')

    
    state = get_data(train = train, segment = 1) 
    
    plot_trend(statelist = statelist, state = state)
    
    #TamilNadu showing highest peak, Removing TamilNadu from top7
    statelist = ['MAHARASHTRA',
                 'WEST BENGAL',
                 'GUJARAT',
                 'UTTAR PRADESH',
                 'PUNJAB',
                 'ANDHRA PRADESH']
    plot_trend(statelist = statelist, state = state)
    
    #WestBengal showing highest peak, Removing it also from top7
    statelist = ['MAHARASHTRA',
                 'GUJARAT',
                 'UTTAR PRADESH',
                 'PUNJAB',
                 'ANDHRA PRADESH']
    plot_trend(statelist = statelist, state = state)
    case = state.groupby(['application_date'])['case_count'].mean().to_numpy()
        
    plt.plot(case)
    median_case_calendar, ten_day_medians = median_value(case = case,window = 10)    
    
    plt.plot(ten_day_medians)
    
    plt.plot(tools.autocorrelation(case= case))

    for test_day in range(0,31):
        test_year = 2019
        test_month = 7
        prediction = predict(test_year,test_month,test_day, median_case_calendar)
    
        
    # plot of case with 10 day median        
    plt.scatter(ten_day_medians,case, alpha =0.3 )      
    # Most of value lies between 0-50 and median between 20-40
    # spike between  20-40, For median 40 : 240
     
    # plot of median with residual        
    plt.scatter(ten_day_medians,ten_day_median -case, alpha =0.3  )
    # higher error for 20 to 40 , especially 30-40 : lower confidence
    
# hmm 
# K filter

# state decomposition models

# Different model for segment 1 and 2 
# MAPE : error function
# library : Profit  
# Feature engineering time series
# Brendon Rohrer
# Decomposition 
# Plot     

# Feature engineering 
# Correlation between states    
    
    






import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
mod = sm.tsa.statespace.SARIMAX(train,
                                    order = (2, 0, 4),
                                    seasonal_order = (3, 1, 2, 12),
                                    enforce_stationarity = False,
                                    enforce_invertibility = False)
results = mod.fit()
results.plot_diagnostics(figsize=(15,12))
plt.show()