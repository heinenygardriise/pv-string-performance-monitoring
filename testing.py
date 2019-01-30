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
    [currentDf, voltageDf, misc_df] = pickle.load(pickle_in)
    
currentDf = currentDf.resample('10T').median()
voltageDf = voltageDf.resample('10T').median()
misc_df = misc_df.resample('10T').median()


   

#%% Testing filtering algorithm
# Illustrating the difference between the different clearsky filtering
plt.close('all')

latitude = 30.144921 #   
longitude = 35.821390 #   
altitude = 800 # moh.
tz = 'Asia/Damascus'

pspm_df, map_df = pspm.get_string_index_map(currentDf, tagConvention='SMA')


use_indexes2, clearsky2, clearTimes2 = pspm.filter_before_aggregate(
                                        misc_df.GHI, latitude, longitude, tz, 
                                        altitude, clearsky_filter=False, filter_by_hours=4,
                                        GHI_cutoff=100)

use_indexes3, clearsky3, clearTimes3, detect_cs_components = pspm.filter_before_aggregate(
                                            misc_df.GHI, latitude, longitude, tz, 
                                            altitude, clearsky_filter=True, filter_by_hours=4,
                                            GHI_cutoff=100)

plt.figure(figsize=(16,5))
plt.plot(pspm_df.index, pspm_df.iloc[:,1],'k', linewidth=.5)
plt.scatter(pspm_df.loc[use_indexes2].index, pspm_df.loc[use_indexes2,1], marker='x')
plt.scatter(pspm_df.loc[use_indexes3].index, pspm_df.loc[use_indexes3,1], marker='o', facecolors='none', edgecolors='r')

plt.figure(figsize=(16,5))
plt.plot(misc_df.index, misc_df.GHI,'k', linewidth=.5)
plt.scatter(pspm_df.loc[use_indexes2].index, misc_df.loc[use_indexes2,'GHI'], marker='x')
plt.scatter(pspm_df.loc[use_indexes3].index, misc_df.loc[use_indexes3,'GHI'], marker='o', 
            facecolors='none', edgecolors='r')


use_indexes3.loc[misc_df.Tmod.isna() | misc_df.POAI.isna()] = False

daily_QPR, daily_H_poa = pspm.Quasi_PR(pspm_df, misc_df.POAI, misc_df.Tmod, -0.00045, 'D')
daily_QPR3, daily_H_poa3 = pspm.Quasi_PR(pspm_df.loc[use_indexes3], misc_df.loc[use_indexes3,'POAI'], 
                                         misc_df.loc[use_indexes3,'Tmod'], -0.00045, 'D')

fig, ax = plt.subplots(figsize=(16,5))
#ax.scatter(daily_QPR.index, daily_QPR.iloc[:,1])
#ax.scatter(daily_QPR3.index, daily_QPR3.iloc[:,1], marker='^')

daily_QPR_filter = pspm.filter_after_aggregate(daily_QPR)
daily_QPR_filter3 = pspm.filter_after_aggregate(daily_QPR3)
ax.scatter(daily_QPR_filter.index, daily_QPR_filter.iloc[:,1])
ax.scatter(daily_QPR_filter3.index, daily_QPR_filter3.iloc[:,1], 
            marker='^', facecolors='none', edgecolors='r')







#%% Call string_comparison

#[trigger_string_performance, string_performance, map_df] = pspm.compare_strings(
#        currentDf.loc['2018-05'])






#%% Find limits of performance    

#pspm.find_limits_of_performance_naive(currentDf)


#plt.scatter(daily_QPR.index, daily_QPR.iloc[:,1])



#metrics = pspm.historical_relative_performance_per_string(daily_QPR)
