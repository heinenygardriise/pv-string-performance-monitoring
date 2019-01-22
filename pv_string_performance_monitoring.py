# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:19:02 2019

@author: asmunds
"""

import pandas as pd
import numpy as np

def compare_strings(input_df):
    ''' 
    Compare strings of a pv system by current/power
    Inputs:
        - input_df: currents or powers of different strings
        
    returns:
        - trigger_string_performance: True if string performance < i'th percentile,
            where i are column names
        - string_performance: gives deviation from median and mean in the period of each string
    '''
    
    percentiles = [1,2,3,4,5,10,20,30,40,45]
    
    
    aggregates = input_df.sum(axis=0)
    meanProd   = aggregates.mean()
    medianProd   = aggregates.median()
    
    string_performance = pd.DataFrame(index=input_df.columns, columns=['dev_from_mean',
                                      'dev_from_median'], data=np.nan)
    trigger_string_performance = pd.DataFrame(index=input_df.columns, 
                                              columns=percentiles)
    
    string_performance['dev_from_mean'] = aggregates-meanProd
    string_performance['dev_from_median'] = aggregates-medianProd
    
    for i in percentiles: 
        trigger_string_performance[i] = aggregates < aggregates.quantile(i/100)
    
    return [trigger_string_performance, string_performance]


    
    