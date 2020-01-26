import numpy as np

def find_day_of_year(year, month, day):
    '''
    Parameters:
     ---------------
        year : int
        month : int
        day : int


    Returns:
     ---------------
        day_of_year: array of integers

    '''
    days_per_month = np.array(
        [31,  # January
         28,  # Feb
         31,  # Mar
         30,  # Apr
         31,  # May
         30,  # June
         31,  # July
         31,  # August
         30,  # Sep
         31,  # Oct
         30,  # Nov
         31,  # Dec
         ])

    if year % 4 == 0:
        days_per_month[1] += 1

    day_of_year = np.sum(days_per_month[:month - 1]) + day - 1

    return day_of_year

def autocorrelation(case = None):
    """
    finds autocorrelation
    
    Parameters:
        case = array of float
    -----------
    
    
    Return:
    -----------
    autocorr : array of float
        autocorrelation till a lag of 1000
    
    """    
    autocorr=[]
    for shift in range(1,1000):
        correlation=np.corrcoef(case[shift:],case[:-shift])[1,0]
        autocorr.append(correlation)
    return autocorr



# Perform Dickey - Fuller test
def test_Seasonality(indexedDataset):
    
    """
    Check seasonality using two methods : Rolling Mean and ADF
    
    Parameters:
    ----------
    
    indexedDataset : Pandas Series with index as time
    
    """
        
    # Determine rolling statistics
    rolmean = indexedDataset.rolling(window = 365).mean()
    rolstd = indexedDataset.rolling(window = 365).std()
    
    # Plot rolling statistics
    orig = plt.plot(indexedDataset,color = 'blue',label = 'Original')
    mean = plt.plot(rolmean,color = 'red',label = 'Rolling Mean')
    std = plt.plot(rolstd,color = 'black',label = 'Rolling Std')
    plt.legend(loc ='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block = False)

    
    print('Results of Dicket - Fuller Test')
    dftest = adfuller(indexedDataset['case_count'],autolag ='AIC')
    dfoutput = pd.Series(dftest[0:4],index = ['Test Statistics','p-value','#Lags Used','Number of Observation Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# Estimating Trend                     
    
def plot_estimate_trend(indexedDataset):
    
        
    """
    Plots trend, seasonality, residual 
    
    Parameters:
    ----------
    
    indexedDataset : Pandas Series with index as time
    
    """

    decomposition = seasonal_decompose(indexedDataset, freq = 3) 
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    plt.subplot(411)
    plt.plot(indexedDataset, label = 'Original')
    plt.legend(loc ='best')
    
    plt.subplot(412)
    plt.plot(trend, label = 'Trend')
    plt.legend(loc ='best')
    
    plt.subplot(413)
    plt.plot(seasonal, label = 'Seasonal')
    plt.legend(loc ='best')
    
    plt.subplot(414)
    plt.plot(residual, label = 'Residual')
    plt.legend(loc ='best')
    
    plt.tight_layout()

# ACF and PACF
def plot_acf_pac(indexedDatset):
    
    
    """
    Plots ACF, PACF 
    
    Parameters:
    ----------
    
    indexedDataset : Pandas Series with index as time
    
    """    
    lag_acf = acf(indexedDataset,nlags= 20)
    lag_pacf = pacf(indexedDataset,nlags= 5, method ='ols')
    
    #Plot ACF
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color ='gray')
    plt.axhline(y=1.96/np.sqrt(len(indexedDataset)),linestyle='--',color = 'gray')
    plt.axhline(y=-1.96/np.sqrt(len(indexedDataset)),linestyle='--',color = 'gray')
    plt.title('Autocorrelation Function')
    
    #Plot PACF
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color ='gray')
    plt.axhline(y=1.96/np.sqrt(len(indexedDataset)),linestyle='--',color = 'gray')
    plt.axhline(y=-1.96/np.sqrt(len(indexedDataset)),linestyle='--',color = 'gray')
    plt.title('Partial Autocorrelation Function')
    
    plt.tight_layout() 



import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100