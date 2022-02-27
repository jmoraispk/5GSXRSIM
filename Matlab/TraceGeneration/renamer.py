# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 12:25:03 2021

@author: Morais
"""

import os
path = r'C:\Zheng Data\TU Delft\Thesis\Thesis Work\GitHub\SXRSIMv3\Matlab' + \
       r'\TraceGeneration\SEED21_omni_8\Channel_parts'
copy_path = path + r'\copy'
new_files = os.listdir(copy_path)
og_files = os.listdir(path)



for index, file in enumerate(reversed(new_files)):
    if len(file.split('.')) > 1:
        if file.split('.')[1] == 'bin':
            ue_idx = int(file.split("_")[2])
            new_idx = ue_idx*2 - 1
            parts = file.split('_')
            new_file_name = parts[0] + '_' + parts[1] + \
                            f'_{new_idx}_' + \
                            parts[3] + '_' + parts[4]
            os.rename(os.path.join(copy_path, file), 
                      os.path.join(copy_path, new_file_name))

            # print(file)  
            print(new_file_name)

for index, file in enumerate(reversed(og_files)):
    if len(file.split('.')) > 1:
        if file.split('.')[1] == 'bin':
            ue_idx = int(file.split("_")[2])
            new_idx = ue_idx*2
            parts = file.split('_')
            new_file_name = parts[0] + '_' + parts[1] + \
                            f'_{new_idx}_' + \
                            parts[3] + '_' + parts[4]
            os.rename(os.path.join(path, file), 
                      os.path.join(path, new_file_name))
    
#             print(file)  
            print(new_file_name)
    
    # if file.split('_')[-1][0:4] == 'SEED':
    #     print('hey')
    #     new_file_name = 'SEED' + file.split('_')[-1][4:] + '_omni'
    