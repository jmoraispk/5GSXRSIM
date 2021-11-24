# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:01:38 2021

@author: Sandra Dheeraj
"""
import time
import pandas as pd
import numpy as np
data_to_save = sim_data_trimmed[0][13][:,:,0] # sim_data_trimmed[f][4][:,:,0]
# data_to_save = (10 * np.log10(sim_data_trimmed[f][13][:,:,0]))
n_ues = 4
n_ttis = len(trim_ttis)
for ue in range(n_ues):
    for tti in range(0, n_ttis-1):
        if sim_data_trimmed[0][15][tti][ue] == 0: # if ue is not scheduled
            data_to_save[tti, ue] = np.nan
    
#for ue in range(n_ues):
#    print(np.nanmean(data_to_save[:, ue]))

np.savetxt(f"debug_time={time.time()}.csv", data_to_save, delimiter=",")
#%%
# Read the csv as a Pandas dataframe
test = pd.read_csv('C:\\Users\\Srijan\\Desktop\\New.csv')

#Find the top 4 largest
first = test.T.apply(lambda x: x.nlargest(1).idxmin())
second = test.T.apply(lambda x: x.nlargest(2).idxmin())
third = test.T.apply(lambda x: x.nlargest(3).idxmin())
fourth = test.T.apply(lambda x: x.nlargest(4).idxmin())

#Put them in a single dataframe
frames = [first, second, third, fourth]
result = pd.concat(frames, axis = 1)

#save result to csv
result.to_csv('C:\\Users\\Srijan\\Desktop\\result.csv')