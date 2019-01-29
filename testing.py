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


#plt.style.use('ggplot')


#%% Load data

sitename = 'GLAE'
saveFolderName = r"C:\Users\asmunds\OneDrive - Institutt for Energiteknikk\pv-string-performance-monitoring\Testing data\\"
pickleName = sitename+'_data_everything.pickle'
with open(saveFolderName+pickleName,"rb") as pickle_in:
    [currentDf, voltageDf, misc_df] = dfDummy = pickle.load(pickle_in)
    
currentDf = currentDf.resample('10T').median()
voltageDf = voltageDf.resample('10T').median()
misc_df = misc_df.resample('10T').median()



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
    

#%% Testing filtering algorithm
# Illustrating the difference between the different clearsky filtering
plt.close('all')

latitude = 30.144921 #   
longitude = 35.821390 #   
altitude = 800 # moh.
tz = 'Asia/Damascus'

pspm_df, map_df = pspm.get_string_index_map(currentDf)

pspm_df2, GHI_2, clearsky2, clearTimes2 = pspm.filter_before_aggregate(
                                        pspm_df, misc_df.GHI, latitude, longitude, tz, 
                                        altitude, clearsky_filter=False, filter_by_hours=4,
                                        GHI_cutoff=100)

pspm_df3, GHI_3, clearsky3, clearTimes3, components = pspm.filter_before_aggregate(
                                            pspm_df, misc_df.GHI, latitude, longitude, tz, 
                                            altitude, clearsky_filter=True, filter_by_hours=4,
                                            GHI_cutoff=100)

plt.figure(figsize=(16,5))
plt.plot(pspm_df.index, pspm_df.iloc[:,1],'k', linewidth=.5)
plt.scatter(pspm_df2.index, pspm_df2.iloc[:,1], marker='x')
plt.scatter(pspm_df3.index, pspm_df3.iloc[:,1], marker='o', facecolors='none', edgecolors='r')

plt.figure(figsize=(16,5))
plt.plot(misc_df.index, misc_df.GHI,'k', linewidth=.5)
plt.scatter(GHI_2.index, GHI_2, marker='x')
plt.scatter(GHI_3.index, GHI_3, marker='o', facecolors='none', edgecolors='r')









#daily_QPR, daily_H_poa = pspm.Quasi_PR(pspm_df, misc_df.POAI, misc_df.GHI, 'D')
#
#daily_QPR = pspm.filter_after_aggregate(daily_QPR)


#%% Find limits of performance    

#pspm.find_limits_of_performance_naive(currentDf)


#plt.scatter(daily_QPR.index, daily_QPR.iloc[:,1])



#metrics = pspm.historical_relative_performance_per_string(daily_QPR)
