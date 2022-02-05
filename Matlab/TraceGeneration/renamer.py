# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 12:25:03 2021

@author: Morais
"""

import os
path = r'C:\Users\Morais\Documents\SXR_Project\SXRSIMv3\Matlab\TraceGeneration'
files = os.listdir(path)


for index, file in enumerate(files):
    if file.split('_')[-1][0:4] == 'SEED':
        print('hey')
        new_file_name = 'SEED' + file.split('_')[-1][4:] + '_omni'
        os.rename(os.path.join(path, file), os.path.join(path, new_file_name))
    