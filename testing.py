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
pickleName = sitename+'_data_month.pickle'
with open(saveFolderName+pickleName,"rb") as pickle_in:
    [currentDf, voltageDf, misc_df] = dfDummy = pickle.load(pickle_in)
    

#%% Call string_comparison

[trigger_string_performance, string_performance] = pspm.compare_strings(
        currentDf.loc['2018-01-02'])

