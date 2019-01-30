# -*- coding: utf-8 -*-
"""
pv_string_performance_monitoring.py
Created on Thu Jan 17 12:19:02 2019

 - A library of functions for PV string performance analytics

Functions:
    - get_string_index_map: returns datframe with numbered columns and a mapping of column names to column numbers
    - Quasi_PR: returns the daily sum of the input columns divided by the daily sum of the irradiance
    - filter_after_aggregate: removes outliers and fluctuations
    - filter_before_aggregate: filters away 0-values, optional clearsky-filtering, optinal filtering 
        on # of hours around noon
    - compare_strings: Compare strings of a pv system by current/power


@author: asmunds
"""

import pandas as pd
import numpy as np
import pvlib





def get_string_index_map(current_df, tagConvention=None):
    ''' 
    Maps string names to column numbers
    Returns input_df with numbered columns and map_df with the mapping between the column numers
    and the old column names 
    (as well as the option of reading current channel #, SM # and inverter # for a specific tag
    convention)
    '''
    currentCols = current_df.columns
    current_df = current_df.copy()
    map_df = pd.DataFrame(index=range(len(current_df.columns)), data={'string_names': currentCols})
    current_df.columns = range(len(current_df.columns))
    
    if tagConvention == 'SMA':
        map_df = pd.concat([map_df, 
                            pd.DataFrame(index=map_df.index, columns=['Inverter','SM','Channel'])])
        for i,col in enumerate(currentCols):
            map_df.loc[i,'Channel'] = int(col[-2:])
            map_df.loc[i,'SM'] = int(col[-20])
            map_df.loc[i,'Inverter'] = int(col[-25:-23])

    return  current_df, map_df





def Quasi_PR(input_df, POAI, Tmod=0, Tcoeff=0, frequency='D'):
    ''' 
    Calculates the input_df (which can be a current or a power) divided by the plane of array 
    irradiance and returns this resampled by the frequency (e.g. 'D' (daily), 'M' (monthly), etc.). 
    
    Also return H_poa, which is the daily irradiation in the plane of array (in Wh, if the POAI is 
    in W)
    '''
    print('Calculating Quasi PR')
    input_freq = 60/np.round(int(np.mean(POAI.index[1:] - POAI.index[:-1]))/60e9)
    try:
        QPR = input_df.resample(frequency).sum().divide(
                (POAI*(1+Tcoeff*(Tmod-25))).resample(frequency).sum(), axis='index')
        H_poa = POAI.resample(frequency).sum()/input_freq
        return QPR, H_poa

    except:
        pass
  
    
    
    
def filter_after_aggregate(QPR):
    ''' 
    A general filter for removing features irrelevant to string performance monitoring
    (after aggregating)
    '''
    print('Filter after aggregate..')
    QPR=QPR.copy()
        
    # Remove obvious outliers
    Q1 = QPR.quantile(.1)
    Q9 = QPR.quantile(.9)
    QPR[QPR>2*Q1] = np.nan
    QPR[QPR<Q9/2] = np.nan
    
    # Remove "fluctuations"
    std = QPR.std()
    QPR[np.abs(QPR.ffill()-QPR.ffill().rolling(7, center=True).median())>std] = np.nan
    
    return QPR






def filter_before_aggregate(GHI, latitude, longitude, tz, altitude, clearsky_filter=True,
                            filter_by_hours=False, GHI_cutoff=500):
    ''' 
    A filter for filtering for clearsky (optional) and remain with the [filter_by_hours] hours 
    around solar noon (optional)
    If filter_by_hours=False: Keep all hours
    If filter_by_hours = x (where x is an integer): Filter for x hours around solar noon
    (before aggregating)
    
    Returns: 
     - use_indexes, which is a pandas.Series with boolean data, True for indexes we'd like to keep
     - clearsky, which is a pandas.DataFrame with GHI, DHI and DNI from pvlib
     - clearTimes, which is a pandas.Series with boolean data, True for indexes with clear skies
     - detect_cs_components (optional), which are the components of the detect_clearsky from pvlib
    
    The clearsky filtering has been optimized for data with 10-minute intervals
    '''
    use_indexes = pd.Series(index=GHI.index, data=True)
    mean_interval = np.round(int(np.mean(GHI.index[1:] - GHI.index[:-1]))/60e9)

    GHI_freq = GHI.index.freq
    if GHI_freq:
        if ~isinstance(GHI_freq, str):
            GHI_freq = GHI_freq.freqstr
        GHI[GHI.isna()] = 0
        GHI = GHI.resample(GHI_freq).mean()
    else:
        print('GHI needs to have a frequency. Clearsky filtering was not possible')
        clearsky_filter=False
    
    use_indexes.loc[GHI<GHI_cutoff] = False
#    input_df[input_df<=0] = np.nan
    
    print('Calculating clear sky irradiance')
    Location = pvlib.location.Location(latitude, longitude, tz, altitude)
    clearsky = Location.get_clearsky(GHI.index)
    
    if clearsky_filter==True:
        print('Clearsky filtering...')
        window_length = mean_interval*6
        GHI_frame = pd.DataFrame(GHI)
        GHI_frame['Date'] = GHI_frame.index.date
        mean_diff = GHI_frame.groupby('Date').mean().values.std()
        clearTimes, detect_cs_components, alpha = pvlib.clearsky.detect_clearsky(
                                                measured=GHI, 
                                                clearsky=clearsky['ghi'], 
                                                times=GHI.index, 
                                                window_length=window_length,
                                                mean_diff=mean_diff*2,
                                                max_diff=mean_diff*2,
                                                lower_line_length=-mean_diff*mean_interval,
                                                upper_line_length=mean_diff*mean_interval,
                                                var_diff=mean_diff/2,
                                                slope_dev=mean_diff/2,
                                                max_iterations=50,
                                                return_components=True)
#        # For tuning of parameters
#        for key in components.keys():    
#            print(key, np.sum(components[key]))
#        print(alpha)
#        print(clearTimes)
        
    else: # or by fraction of clearsky (and hours)
        clearTimes = (GHI > 0.9*clearsky['ghi'])

    use_indexes.loc[~clearTimes] = False

    if filter_by_hours:
        print('Filter by hours...')
        solarPosition = Location.get_solarposition(GHI.index)
        solarPosition['Date'] = solarPosition.index.date

        solarNoonByDate = pd.Series(index=solarPosition.groupby('Date').min().index,
                            data=[solarPosition.loc[solarPosition['Date']==date,'zenith'].idxmin()
                                  for date in solarPosition.groupby('Date').min().index])
        solarNoon = pd.Series(index=GHI.index, data=[solarNoonByDate.loc[date] for date 
                                                          in GHI.index.date])
        timeDiffs = np.abs(GHI.index-solarNoon)
        use_indexes.loc[timeDiffs>pd.Timedelta(str(filter_by_hours/2)+' hours')] = False
                
    if clearsky_filter:
        return use_indexes, clearsky, clearTimes, detect_cs_components
    else:
        return use_indexes, clearsky, clearTimes







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