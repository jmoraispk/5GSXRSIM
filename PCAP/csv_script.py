# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 19:05:55 2021

@author: duzhe
"""

import csv
import os 
import sys
import numpy as np
def main(packets, droprate):
    # Nr. total packets
    # Desired droprate in %
    data = np.zeros(packets)
    header = 'List'
    for packet in range(1, packets+1): 
        if packet % int(100/droprate) != 0: 
            data[packet-1] = 1 # print(packet, 'successful')
        else: 
            pass # data[packet] = 0    # print(packet, 'dropped')
    
    output_save_path = 'C:\Zheng Data\TU Delft\Thesis\Thesis Work\GitHub\SXRSIMv3\PCAP\PDR_output'
    output_file_name = f'PDR_{droprate}%.csv'

    complete_file_name = os.path.join(output_save_path, output_file_name)
    
    
    np.savetxt(complete_file_name, data, delimiter=",", fmt='%s')
    
    print('Saved to', complete_file_name)
    return

nr_packets = 19472 
# pdr = [0.1,0.5,1,2,3,4,5] # in percent
pdr = [1.5,2.5,3.5,4.5]
for i in range(len(pdr)):
    main(nr_packets, pdr[i])

print('done')
    
    