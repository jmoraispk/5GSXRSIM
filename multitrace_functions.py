# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 09:21:48 2021

@author: Morais
"""


import numpy as np
import utils as ut
import matplotlib.pyplot as plt


###########################  Multi Trace Functions  ##########################


def plot_f(x_vals, sim_data, ue, file_idxs, var_idxs):
    """
    Joins in the same plot the data from multiple, for a single user
    """
    
    for f in file_idxs:
        for v in var_idxs:
            plt.plot(x_vals, sim_data[v][f][:,ue], label=f'File{f}, var{v}')
            
    plt.legend()
    plt.show()


def plot_f2(ue_list, x_vals, sim_data, file_idxs, var_idxs, 
            x_axis_label='', y_axis_label='',
            title='', linewidths='',y_labels='', use_legend=False,
            legend_inside=False, legend_loc="center", 
            legend_coords=(0.53, -0.01), ncols=1, size=1, width=6.4, 
            height=4.8, filename='', savefig=False, double_mode=False,
            uniform_scale=[], same_axs=False):
    
    if filename == '':
        filename = title
    
    r = width/height
    
    if same_axs:
        fig, axs = plt.subplots(tight_layout=True, 
                                figsize=(r*height*size, size/r*width))
    else:
        n_ue = len(ue_list)
        if n_ue > 1:
            div_list = ut.divisors(n_ue)
            n1 = div_list[1]
            n2 = div_list[-2]
        else:
            n1 = 1
            n2 = 1
        
        fig, axs = plt.subplots(n1,n2, tight_layout=True, 
                                figsize=(r*height*size, size/r*width))
    
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    
    for ue in ue_list:
        
        if same_axs:
            idx = 0
        else:
            if n2 > 1:
                aux = int(len(ue_list) / 2)
                if ue < aux:
                    idx = (0, ue)
                else:
                    idx = (1, ue - aux)
            else:
                idx = ue_list.index(ue)
                
        ax_handle = axs[idx]
        
        
        for v in var_idxs:
            for f in range(len(sim_data[v])):
            # Before was: for f in file_idxs:
            # But this way we can add averages to the sim_data and plot those 2
                if y_labels == '':
                    ax_handle.plot(x_vals, sim_data[v][f][:,ue], 
                                   label=f'File{f}, var{v}')
                else:
                    ax_handle.plot(x_vals, sim_data[v][f][:,ue], 
                                   label=y_labels[f])
                    
                    
            # ax_handle.plot(x_vals, y_vals[y_idx][:, ue], alpha=opacity, 
            #                linewidth=linewidths[p_idx], label=y_labels[p_idx])

        ax_handle.set_xlabel(x_axis_label)
        ax_handle.set_xlabel(x_axis_label)
        ax_handle.set_ylabel(y_axis_label)
        ax_handle.set_title(f'UE {ue}')
        
        if uniform_scale:
            if isinstance(uniform_scale, bool):
                # Get the biggest limit and set all others accordingly
                # Getting the biggest limit needs to be done previously
                # here we just set
                pass
            
            
            ax_handle.set_ylim(uniform_scale)
        
        #ax_handle.set_xlim([min(x_vals)-1, max(x_vals)+1])
        #ax_handle.autoscale(enable=True, axis='x', tight=True)
        #ax_handle.autoscale(enable=True, tight=True)
        
        if legend_inside:
            legend_handle = ax_handle.legend(loc=legend_loc)
        
    if use_legend and not legend_inside:
        
        handles, labels = ax_handle.get_legend_handles_labels()
        # loc sets the point of the box to anchor. 
        # Selecting 'center' puts the centre of the box when we say, in 
        # relation to the figure origin. Negative values can be used for 
        # the bounding box
        legend_handle = fig.legend(handles, labels, loc=legend_loc, 
                                   bbox_to_anchor=legend_coords,
                                   fancybox=True, shadow=True, ncol=ncols)
        
    if use_legend:
        for legobj in legend_handle.legendHandles:
            legobj.set_linewidth(2.0)
    
    # Subplot adjusting for more creative plots: 
    # https://stackoverflow.com/questions/6541123/
    
    if title != '':
        fig.suptitle(title)
    
    if filename != '' and savefig:
        if use_legend and not same_axs:
            fig.savefig(filename + '.pdf', format='pdf',
                        bbox_extra_artists=(legend_handle,), 
                        bbox_inches='tight')
        else:
            plt.savefig(filename + '.pdf', format='pdf')
        
        print(f'Saved: {filename}')
    plt.show()

