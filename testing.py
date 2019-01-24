# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 09:30:00 2019

@author: asmunds
"""

import pandas as pd
import numpy as np
import pv_string_performance_monitoring as pspm
import pickle
import matplotlib.pyplot as plt




#%% Load data

sitename = 'GLAE'
saveFolderName = r"C:\Users\asmunds\OneDrive - Institutt for Energiteknikk\pv-string-performance-monitoring\Testing data\\"
pickleName = sitename+'_data_everything.pickle'
with open(saveFolderName+pickleName,"rb") as pickle_in:
    [currentDf, voltageDf, misc_df] = dfDummy = pickle.load(pickle_in)
    
currentDf = currentDf[~currentDf.index.duplicated(keep='first')]
voltageDf = voltageDf[~voltageDf.index.duplicated(keep='first')]
misc_df = misc_df[~misc_df.index.duplicated(keep='first')]



#%% Prepare data

#[currentDf, map_df] = pspm.get_string_index_map(currentDf)



#useIndexes = misc_df.loc[misc_df.POAI>0].index
#PM = pd.DataFrame(index=useIndexes, columns=currentDf.columns)
#for i,col in enumerate(currentDf.columns):
#    if ~i%100: print(i)
#    PM[col] = currentDf.loc[useIndexes,col]/misc_df.POAI.loc[useIndexes]



#%% Call string_comparison

#[trigger_string_performance, string_performance, map_df] = pspm.compare_strings(
#        currentDf.loc['2018-05'])
    

#%% Find limits of performance    

#pspm.find_limits_of_performance_naive(currentDf)


pspm_df, map_df = pspm.get_string_index_map(currentDf)
daily_QPR, daily_H_poa = pspm.Quasi_PR(pspm_df, misc_df.POAI, misc_df.GHI, 'D')

daily_QPR = pspm.filter_after_aggregate(daily_QPR)
plt.figure(figsize=(14,6))
plt.scatter(daily_QPR.index, daily_QPR.iloc[:,1])



#metrics = pspm.historical_relative_performance_per_string(daily_QPR)
