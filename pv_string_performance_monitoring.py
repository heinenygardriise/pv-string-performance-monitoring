# -*- coding: utf-8 -*-
"""
pv_string_performance_monitoring.py
Created on Thu Jan 17 12:19:02 2019

 - A library of functions for PV string performance analytics

Functions:
    - get_string_index_map


@author: asmunds
"""

import pandas as pd
import numpy as np
import pvlib



def get_string_index_map(input_df):
    ''' 
    Maps string names to column numbers
    Returns input_df with numbered columns and map_df with the mapping between the column numers
    and the old column names
    '''
    output_df = input_df.copy()
    map_df = pd.DataFrame(index=range(len(output_df.columns)), data={'string_names': output_df.columns})
    output_df.columns = range(len(output_df.columns))
    
    return  output_df, map_df




def Quasi_PR(input_df, POAI, GHI=None, frequency='D'):
    ''' 
    Calculates a the input_df divided by the plane of array irradiance and returns this 
    resampled by the frequency (e.g. 'D' (daily), 'M' (monthly), etc.). 
    Also return H_poa, which is the daily irradiation in the plane of array (in Wh)
    '''
    input_freq = 60/np.round(int(np.mean(POAI.index[1:] - POAI.index[:-1]))/60e9)
    try:
        QPR = input_df.resample(frequency).sum().divide(POAI.resample(frequency).sum(), axis='index')
        H_poa = POAI.resample(frequency).sum()/input_freq
        return QPR, H_poa

    except:
        pass
  
    
    
def filter_after_aggregate(QPR):
    ''' 
    A general filter for removing features irrelevant to string performance monitoring
    (after aggregating)
    '''
    # Remove obvious outliers
    Q1 = QPR.quantile(.1)
    Q9 = QPR.quantile(.9)
    QPR[QPR>2*Q1] = np.nan
    QPR[QPR<Q9/2] = np.nan
    
    # Remove "fluctuations"
    std = QPR.std()
    QPR[np.abs(QPR.ffill()-QPR.ffill().rolling(7, center=True).median())>std] = np.nan
    
    return QPR




def filter_before_aggregate(input_df, GHI, latitude, longitude, tz, altitude, clearsky_filter=True,
                            filter_4_hours=False):
    ''' 
    A general filter for removing features irrelevant to string performance monitoring
    (after aggregating)
    '''
    
    input_df[GHI<=0] = np.nan
    input_df[input_df<=0] = np.nan
    
    Location = pvlib.location.Location(latitude, longitude, tz, altitude)
    clearsky = Location.get_clearsky(GHI.index)
    
    if clearsky_filter==True:
        print('Clearsky filtering...')
        window_length = GHI.index.freq*6
        GHI['Date'] = GHI.index.date
        mean_diff = GHI.groupby('Date').mean().std()
        max_diff = mean_diff
        slope_dev = mean_diff/2
        upper_line_length = mean_diff/window_length
        lower_line_length = 0
        var_diff = np.inf
        max_iterations = 50
        
        clearTimes = pvlib.clearsky.detect_clearsky(GHI, clearsky['ghi'], GHI.index,
            window_length, mean_diff, max_diff, slope_dev,
            upper_line_length, lower_line_length, var_diff, max_iterations)
        input_df = input_df[clearTimes]
        GHI = GHI[clearTimes]
        input_df = input_df[GHI>500]
    else: # or by fraction of clearsky (and hours)
#        clearsky = clearsky.reindex(input_df.index, method='nearest')
        input_df = input_df[GHI > 0.9*clearsky['ghi']]

    if filter_4_hours:
        solarPosition = Location.get_solarposition(input_df.index)
        solarPosition['Date'] = solarPosition.index.date
        solarNoonByDate = pd.Series(index=solarPosition.groupby('Date').min().index,
                            data=[solarPosition.loc[solarPosition['Date']==date,'zenith'].idxmin()
                                  for date in solarPosition.groupby('Date').min().index])
        solarNoon = pd.Series(index=input_df.index, data=[solarNoonByDate.loc[date] for date 
                                                          in input_df.index.date])
        timeDiffs = np.abs(input_df.index-solarNoon)
        input_df = input_df[timeDiffs<pd.Timedelta('2 hours')]
        
#        SolarNoonTimeDecimal = np.mean(
#                [solarNoon.hour+solarNoon.minute/60
#                 for date in solarPosition.groupby('Date').min().index])
        
#        SolarNoonHour = int(np.round(SolarNoonTimeDecimal))
#        maxHour = SolarNoonHour + 2
#        minHour = SolarNoonHour - 2
#        input_df = input_df[input_df.index.hour<maxHour] # Exclude times too far from solar noon
#        input_df = input_df[input_df.index.hour>minHour]
        
    return input_df



def compare_strings(input_df):
    ''' 
    Compare strings of a pv system by current/power
    
    Inputs:
        - input_df: time series of currents or powers of different strings
        
    returns:
        - trigger_string_performance: True if string performance < i'th percentile,
            where i are column names
        - string_performance: gives deviation from median and mean in the period of each string
        - map_df: A dataframe mapping index # to column names
    '''
    # These are the percentiles we are looking for in the dataset
    percentiles = [1,2,3,4,5,10,20,30,40,45]    
    
    # Sum of productipon of each string
    aggregates = pd.DataFrame(index=input_df.columns, data=input_df.sum(axis=0).values) 
    
    meanProd   = aggregates.mean() # Mean of all sums
    medianProd   = aggregates.median() # Median of all sums
    
    # Initializing...
    string_performance = pd.DataFrame(index=input_df.columns, 
                                      columns=['dev_from_mean', 'dev_from_median'],
                                      data=np.nan)
    trigger_string_performance = pd.DataFrame(index=input_df.columns, 
                                              columns=percentiles)
    
    # Fractional deviation from mean and median
    string_performance['dev_from_mean'] = (aggregates-meanProd)/meanProd 
    string_performance['dev_from_median'] = (aggregates-medianProd)/meanProd

    for i in percentiles: # Set True if aggregate < ith percentile, otherwise False
        trigger_string_performance[i] = aggregates < aggregates.quantile(i/100)
    
#    return [trigger_string_performance, string_performance, map_df]




def historical_relative_performance_per_string(input_df):
    '''
    Looking at historical data, find the the range of the (relative) performance during 
    "normal operation" (no fault) of each string (relative to mean/median). 
    
    Inputs:
        - input_df: (historical) time series of currents or powers of different strings
                    (typically > 1 year; 1 hour resolution (or less))
    Outputs:
        - metrics: pandas.describe() of daily sums of input_df
    '''
    input_df[input_df<=0] = np.nan
    daily_sums = pd.DataFrame(columns=input_df.columns, 
                              data=input_df.resample('D').sum())
    metrics = pd.DataFrame(index=daily_sums.columns,
                           columns=input_df.iloc[0:1].describe().index,
                           data=daily_sums.describe().values.transpose())
    
    return metrics





def find_limits_of_performance_naive(input_df):
    '''
    Looking at historical data, find limits for which the strings sohuld perform within.
    This function is naive, in that it only looks at the input_df, and does not take into account
    any other information (irradiance, temperature, etc.)
    
    Inputs:
        - input_df: (historical) time series of currents or powers of different strings
                    (typically > 1 year; 1 hour resolution (or less))
    
    Temporary: Plotting cumulative distribution function of currents and median current per day
    '''    
#    times_of_zero = input_df.loc[input_df.quantile(.75, axis=1)<=0].index # 75% of strings are 0
    input_df[input_df<=0] = np.nan
    aggregates = pd.DataFrame(columns=['Daily sum of medians'], 
                              data=input_df.median(axis=1).resample('D').sum())
    
#    metrics = pd.DataFrame(index=input_df.columns, 
#                           columns=input_df.iloc[0:1].describe().index,
#                           data=input_df.describe().values.transpose())
    
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.figure(figsize=(14,6))
    quantiles = [aggregates.quantile(i/100).mean() for i in range(100)]
    plt.plot(range(100), quantiles)
    
    plt.figure(figsize=(14,6))
    plt.plot(aggregates)
    
#    limits = pd.DataFrame(index=range(len(input_df.columns)), 
#                          data={'extremely_low_performance_check_data_quality': input_df., 
#                                   'something_is_definitly_wrong', 'low_but_no_sigar'],)

#    return [limits, map_df]









def check_performance_by_digital_twin():
    ''' 
    Check performance through the digital twins on each string
    '''
    pass
    
    
def train_digital_twin():
    ''' 
    Train digitial twins on each string
    '''
    pass



class pm_frame(pd.DataFrame):
    ''' 
    Apparantly, making subclasses of pandas Dataframes is discouraged (see 
    http://pandas.pydata.org/pandas-docs/stable/extending.html#extending-subclassing-pandas)
    
    A class that (potentially) contains methods and variables that will be relevant to PV string
    performance monitoring
    '''
#    def __init__(self, input_df):
#        self = input_df.copy()
        
    def reindex(self):
        ''' 
        Maps string names to column numbers
        Returns input_df with numbered columns and map_df with the mapping between the column numers
        and the old column names
        '''
        map_df = pd.DataFrame(index=range(len(self.columns)), 
                              columns=['string names'], 
                              data=self.columns)
        self.columns = range(len(self.columns))
        return map_df