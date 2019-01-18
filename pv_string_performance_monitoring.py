# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:19:02 2019

@author: asmunds
"""

import pandas as pd
import numpy as np

def compare_strings(current_df):
    ''' Compare strings of a pv system by current '''
    
    aggregates = current_df.sum(axis=0)
    meanProd   = aggregates.mean()
    medianProd   = aggregates.median()
    
    string_performance = pd.DataFrame(index=current_df.columns, columns=['dev_from_mean',
                                      'dev_from_median'], data=np.nan)
    trigger_string_performance = pd.DataFrame(index=current_df.columns, data=np.nan)
    
    string_performance['dev_from_mean'] = aggregates-meanProd
    string_performance['dev_from_median'] = aggregates-medianProd
    
    for i in [1,2,3,4,5,10,20,30,40,45]:
        trigger_string_performance[str(i)] = aggregates < aggregates.quantiles(i/100)
    
    


    
    