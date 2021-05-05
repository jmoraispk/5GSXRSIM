# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:37:21 2021

@author: Morais

Objective: to take the values computed previously across traces (many seeds), 
put them in an array and make plots for them.

"""

import numpy as np
import matplotlib.pyplot as plt

import plots_functions as plt_func


########################### Previous Plots ###############################
folder = r'Results\DirectlyFromSandra' + '\\'
# folder = r'Results' + '\\'


# Speeds
# speed_labels = ['Movement index = 1', 'Movement index = 1', 'Movement index = 5']
speed_labels = ['$\mu$ = 1', '$\mu$ = 3', '$\mu$ = 5']
speed_file = folder + r'Batch 1 - Speeds\speeds_per_18seeds_f[F]_16.0s.csv'


# CSI Periodicities
csi_per_labels = ['1 ms', '2 ms', '5 ms', '10 ms', 
                  '20 ms', '50 ms', '100 ms', '200 ms']
csi_per_file = \
    folder + r'Batch 1 - csi periodicities\CSI_per_20seeds_f[F]_16.0s.csv'

# csi_per_file = \
#     r'Results\Batch 14 - new csi periodicities\results_f[F]_16s.csv'

# Application Bit rates
app_bit_labels = ['25 Mbps', '50 Mbps', '75 Mbps', '100 Mbps', '125 Mbps', 
                  '150 Mbps', '175 Mbps', '200 Mbps']
app_bit_file = folder + \
    r'Batch 1 - Application Bitrates\APPBIT_per_18seeds_f[F]_16.0s.csv'


# Latencies
latencies_labels = ['1 ms', '2 ms', '5 ms', '10 ms', 
                  '20 ms', '30 ms', '40 ms', '50 ms']

latencies_labels = ['5 ms', '10 ms', '20 ms', '30ms', '40ms', '50 ms']
latencies_file = folder + \
    r'Batch 12 - filtered latencies\results_f[F]_16s.csv'

##########################################################################


############################# Parameters #################################

# put [F] in the place of the frequency index
results_file_to_load = speed_file

# Frequencies
f_idxs = [0, 1]

# Number of seeds
n_seeds = 18

# Colors and Labels for bar charts
x_labels = ['3.5 GHz', '26 GHz']
y_label = 'Packet Loss Ratio [%]'
bar_labels = speed_labels    

# Number of values of the varying variable: speeds, csi periodicities, ...
n_vvar = len(bar_labels)

# Barwidth (duh...)
barwidth = 0.1


# change names of drop rate array and so on...

##########################################################################

colors = ['C' + str(i) for i in range(len(bar_labels))]

f_len = len(f_idxs)

if len(x_labels) != f_len:
    print('Warning: it is usual to have one set of bars per frequency.')

# student t ditribution value for confidence interval of 95%
t95 = plt_func.t_student_mapping(n_seeds)

# Row is speed/csi/bitrate/... and column is seed
results = np.zeros([f_len, n_vvar, n_seeds]) 
mean_per_vvar = np.zeros([f_len, n_vvar])
std_per_vvar = np.zeros([f_len, n_vvar])
conf_int = np.zeros([f_len, n_vvar])

# Automatically build results array.

# Assumptions:
#    a) We run ALL the seeds for a given case, and then we switch something, 
#       e.g. the speed. 
#    b) Different frequencies are in different files.
file_parts = results_file_to_load.split('[F]')
for f_idx in f_idxs:
    file_to_load = file_parts[0] + f'{f_idx}' + file_parts[1]
    
    with open(file_to_load, 'r') as fp:
        lines = fp.readlines()
    
    for var_idx in range(n_vvar):
        for seed in range(n_seeds):
            idx = seed + var_idx * n_seeds
            # print(idx)
            line = lines[idx]
        
            num1 = float(line)
            
            # This will read as many numbers as the line has:
            # num1, num2 = map(float, line.split()) # to read 2, etc...
            
            #print(num1)
            results[f_idx, var_idx, seed] = num1

# Compute required statistics
for f_idx in f_idxs:
    for var_idx in range(n_vvar):
        mean_per_vvar[f_idx, var_idx] = np.mean(results[f_idx, var_idx, :])
        # print(mean_per_vvar)
        std_per_vvar[f_idx, var_idx] = np.std(results[f_idx, var_idx, :])
        # print(std_per_vvar)
        conf_int[f_idx, var_idx] = \
            (t95 * std_per_vvar[f_idx, var_idx]) / np.sqrt(n_seeds)
        # print(conf_int)

# Ready... Set...
r_base = np.arange(len(x_labels))

# plt.figure(figsize=(8, 6))

# ...Plot!
for bar_idx in range(n_vvar):
    r = r_base + barwidth * bar_idx
    plt.bar(r, mean_per_vvar[:,bar_idx], yerr=conf_int[:,bar_idx],
            width=barwidth, color=colors[bar_idx], label=bar_labels[bar_idx])


plt.ylabel(y_label)
central_offset = n_vvar * (1 / 2 - barwidth - barwidth/2 - 0.015)
plt.xticks([r + central_offset * barwidth for r in range(len(x_labels))], 
           x_labels)
plt.legend(ncol=1)
plt.tick_params('both')
y_tcks = [0,4,8,12,16,20]
# y_tcks = [0,5, 10, 15, 20]
plt.ylim([y_tcks[0], y_tcks[-1]])
plt.yticks(y_tcks)
# plt.autoscale(enable=True, axis='x', tight=True)

# fs = 13
# fs_leg = 12
# plt.ylabel(y_label, fontsize=fs)
# central_offset = n_vvar * (1 / 2 - barwidth + 0.036)
# plt.xticks([r + central_offset * barwidth for r in range(len(x_labels))], 
#            x_labels, fontsize=fs)
# plt.legend(ncol=2, fontsize=fs_leg)
# plt.tick_params('both', labelsize=fs)


plt.savefig(r'Results Phase 2/speeds.svg', bbox_inches = 'tight')