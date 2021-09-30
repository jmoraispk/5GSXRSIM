# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:01:38 2021

@author: Sandra Dheeraj
"""
import pandas as pd
# import numpy as np

# np.savetxt("rec_power_L4.csv", sim_data_trimmed[f][10][:,:,0], delimiter=",")

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