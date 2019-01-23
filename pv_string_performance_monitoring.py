# -*- coding: utf-8 -*-
"""
pv_string_performance_monitoring.py
Created on Thu Jan 17 12:19:02 2019

 - A library of functions for PV string performance analytics
    
@author: asmunds
"""

import pandas as pd
import numpy as np



def get_string_index_map(input_df):
    ''' 
    Maps string names to column numbers
    Returns input_df with numbered columns and map_df with the mapping between the column numers
    and the old column names
    '''
    map_df = pd.DataFrame(index=range(len(input_df.columns)), data={'string_names': input_df.columns})
    input_df.columns = range(len(input_df.columns))
    return [input_df, map_df]



class pm_frame():
    ''' 
    A class that (potentially) contains methods and variables that will be relevant to PV string
    performance monitoring
    '''
    def __init__(self, input_df):
        ''' 
        Maps string names to column numbers
        Returns input_df with numbered columns and map_df with the mapping between the column numers
        and the old column names
        '''
        self.df = input_df
        self.map_df = pd.DataFrame(index=range(len(input_df.columns)), data={'string_names': input_df.columns})
        self.df.columns = range(len(input_df.columns))
    
    




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
    '''
    input_df[input_df<=0] = np.nan
    print(input_df.iloc[:,1].dropna())
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
