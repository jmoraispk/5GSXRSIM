# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:43:05 2022

@author: duzhe
"""

import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import time

pd.options.mode.chained_assignment = None  # default='warn'


"""
Scripts to use with statistics of pcap traces from output of simulator

- Use script to automate plots with PDR statistics saved as csv/txt/pd as input

"""

stats_path = os.getcwd() + "\\PDR\\" 

# Parameters for plots: 
E2E_budget = [100, 25, 50, 100]# (25) - 50 - 100 [ms] 

bitrate = [100, 50, 100, 150, 200] # "APP100" # 50 - 100 - 150 - 200 [Mbps]

dispersion = [0.6, 0.99] # "0.6" # 0.6 - 0.99 -> Percent of Interframe time 

bw = [125, 100, 125, 150, 200] # "BW-125" # (75) - 100 - 125 - 150 - 200 [MHz]

# IF RAN: 10 - 20 - 30 - 40 - 50 - 60 - 70 - 80 - 90 - 100 [ms]
# IF E2E: SAME AS E2E_budget!
lat = [70, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # "RAN-LAT-100" 

sync_offset = [1.0, 0.0] # "Offset-1.0" # 0.0 - 1.0 (sync vs max. async)

queues = [10, 5, 10, 15] #"10" # 5 - 10 - 15

bg_load = [70.0, 50.0, 70.0, 85.0] # "70.0%" # 50.0 - 70.0 - 85.0 [%]

ran_scheduler = ["PF", "M-LWDF", "Frametype", "EDD"] # PF - M-LWDF - Frametype - EDD   
e2e_scheduler = ["M-LWDF", "Frametype", "EDD"] # M-LWDF - Frametype - EDD   


# Plot 1:
# Vary RAN latency budget - Rest Default parameters










