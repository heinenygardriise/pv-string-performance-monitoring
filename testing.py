# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 09:30:00 2019

@author: asmunds
"""

import pandas as pd
import numpy as np
import pv_string_performance_monitoring as pspm
import pickle




#%% Load data
sitename = 'GLAE'
folderName = "C:\\Users\\asmunds\\Documents\\IPN O&M\\Plant data\\"+sitename+'\\'
saveFolderName = r"C:\Users\asmunds\Documents\IPN O&M\pv-string-performance-monitoring\Testing data\\"
pickleName = sitename+'_data_everything.pickle'
with open(saveFolderName+pickleName,"rb") as pickle_in:
    [currentDf, voltageDf, misc_df] = dfDummy = pickle.load(pickle_in)
    
currentDf = currentDf[~currentDf.index.duplicated(keep='first')]
voltageDf = voltageDf[~voltageDf.index.duplicated(keep='first')]
misc_df = misc_df[~misc_df.index.duplicated(keep='first')]

#%% Prepare data
#[trigger_string_performance, string_performance, map_df] = pspm.compare_strings(
#        currentDf.loc['2018-05'])

useIndexes = misc_df.loc[misc_df.POAI>0].index
PM = pd.DataFrame(index=useIndexes, columns=currentDf.columns)
for i,col in enumerate(currentDf.columns):
    if ~i%100: print(i)
    PM[col] = currentDf.loc[useIndexes,col]/misc_df.POAI.loc[useIndexes]



#%% Call string_comparison

pspm.find_limits_of_performance_naive(PM)