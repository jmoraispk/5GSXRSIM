# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 14:34:43 2021

@author: Morais
"""


import numpy as np
import matplotlib.pyplot as plt

# Note: there's probably a better way that avoids using the import below,
# see app_trafic_plots done in v2.
import matplotlib.ticker as ticker 

import utils as ut

def plot_for_ues(ue_list, x_vals, y_vals, x_axis_label='', y_axis_label='',
                 title='', linewidths='', y_labels='', use_legend=False,
                 legend_inside=False, legend_loc="center", 
                 legend_coords=(0.53, -0.01), ncols=1, size=1, width=6.4, 
                 height=4.8, filename='', savefig=False, uniform_scale=[], 
                 same_axs=False, plot_type='line', n1=-1, n2=-1):
    """
    Plots values in the y_vals variable, for each UE, across x_vals.

    Note that y_vals should have a UE dimension-> [x_dimension, UE_dimension]
    
    
    
    Valid locations for legend_loc:
        right
        center left
        upper right
        lower right
        best
        center
        lower left
        center right
        upper left
        upper center
        lower center
    
    
    same_axs puts all ues in the same axis:
        - cancels the subtitle on top of the subplot
        - cancels the opacity and uses linewidths
    
    
    
    """
    
    if y_vals[0] is None:
        raise Exception('y_vals only has Nones in each UE...')

    if linewidths == '':
        if same_axs:
            linewidths = [1 for ue in ue_list]
        else:
            linewidths = [1 for y_val in y_vals]
    
    if y_labels == '':
        if same_axs:
            y_labels = [f'UE {ue}' for ue in ue_list]
        else:
            y_labels = ['' for y_val in y_vals]
        
    if filename == '':
        filename = title
    
    r = width/height
    
    if same_axs:
        fig, axs = plt.subplots(tight_layout=True, 
                                figsize=(r*height*size, size/r*width))
    else:
        
        if n1 != -1 and n2 != -1:
            # if n1 and n2 have been defined, use those, after a little check
            if len(ue_list) != n1 + n2:
                raise Exception("Something doesn't add up... N1 and N2.")
        else:
            # derive n1 and n2 from the number of UEs
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
    
    n_y_vals = len(y_vals)
    
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
        
        for y_idx in range(n_y_vals):
            
            if same_axs:
                if len(ue_list) == 1:
                    p_idx = y_idx
                    opacity = 1
                else:
                    p_idx = ue # plot idx
                    opacity = 1
                    # figure out what to do when there is more than 1 ue,
                    # with multiple things to plot in the same plot...
            else:
                p_idx = y_idx
                ax_handle.set_title(f'UE {ue}')
                
                if len(y_vals) == 1:
                    opacity = 1
                else:
                    opacity = 1.2 / n_y_vals
            
            if y_vals[y_idx].ndim == 1:
                y_data = y_vals[y_idx]
            else:
                y_data = y_vals[y_idx][:,ue]
            
            try:
                if plot_type == 'line':
                    ax_handle.plot(x_vals, y_data, alpha=opacity, 
                               linewidth=linewidths[p_idx], label=y_labels[p_idx])
                elif plot_type == 'scatter':
                    ax_handle.scatter(x_vals,y_data)  
                elif plot_type == 'bar':
                    ax_handle.bar(x_vals, y_data)
                else:
                    raise Exception(f'No plot type named "{plot_type}".')
            except Exception as e:
                if type(e) == ValueError:
                    print('ERROR DESCRIPTION:')
                    print(e)
                    print('\nNote: a Value Error here usually happens '
                          'when you forget to put y inside a list: '
                          'plot(ues, x, [y])')
                    
                return
                

        ax_handle.set_xlabel(x_axis_label)
        ax_handle.set_ylabel(y_axis_label)
        
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
    
    if savefig:
        if filename == '':
            filename = ut.get_time()
        if use_legend and not same_axs:
            fig.savefig(filename + '.pdf', format='pdf',
                        bbox_extra_artists=(legend_handle,), 
                        bbox_inches='tight')
        else:
            plt.savefig(filename)
            
        print(f'Saved: {filename}')
    
    plt.show()
    return axs


def plot_for_ues_double(ue_list, x_vals, y_vals_left, y_vals_right,
                        x_label, y_label, title='', linewidths='',
                        limits_ax1=[], limits_ax2=[],
                        no_ticks_ax1=[], no_ticks_ax2=[],
                        label_fonts=[13,13],
                        fill=False, use_legend=False,
                        fill_var='', fill_color='grey', 
                        legend_loc='best',
                        legend_inside=False, fill_label='', 
                        width=6.4, height=4.8, size=1,
                        filename='', savefig=False, plot_type_left='line',
                        plot_type_right='line'):
    """
    Plots values in the y_vals variable, for each UE, across x_vals.

    Note that y_vals should have a UE dimension-> [x_dimension, UE_dimension]
    
    The fill variable fill the y axis when it is > 0.
    
    
    """
    
    
    if linewidths == '':
        linewidths = np.ones([len(y_vals_left) + len(y_vals_right),1])
    
    n_ue = len(ue_list)
    if n_ue > 1:
        div_list = ut.divisors(n_ue)
        n1 = div_list[1]
        n2 = div_list[-2]
    else:
        n1 = 1
        n2 = 1

    r = width/height
    fig, axs = plt.subplots(n1,n2, tight_layout=True,
                            figsize=(r*height*size, size/r*width))
    
    if n1 == n2 == 1:
        axs = [axs]
    
    axs1 = []
    axs2 = []
    
    
    for ue in ue_list:
        
        if n2 > 1:
            aux = int(len(ue_list)/2)
            if ue < aux:
                idx = (0,ue)
            else:
                idx = (1,ue-aux)
        else:
            idx = ue_list.index(ue)
            
        ax1_handle = axs[idx]
        ax2_handle = ax1_handle.twinx()
        for y_idx_left in range(len(y_vals_left)):
            if plot_type_left == 'line':
                ax1_handle.plot(x_vals, y_vals_left[y_idx_left][:, ue], 
                                alpha=0.6, color='g', 
                                linewidth=linewidths[y_idx_left])
            elif plot_type_left == 'scatter':
                ax1_handle.scatter(x_vals, y_vals_left[y_idx_left][:, ue], 
                                   alpha=0.6, color='g', 
                                   linewidth=linewidths[y_idx_left])
            elif plot_type_left == 'bar':
                ax1_handle.bar(x_vals, y_vals_left[y_idx_left][:, ue], 
                               alpha=0.6, color='g', 
                               linewidth=linewidths[y_idx_left])
            else:
                raise Exception(f'No plot type named "{plot_type_left}" '
                                 'for left plot.')
            
        for y_idx_right in range(len(y_vals_right)):
            tot_idx = y_idx_right + len(y_vals_left)
            if plot_type_right == 'line':
                ax2_handle.plot(x_vals, y_vals_right[y_idx_right][:, ue], 
                            alpha=0.6, color='b', 
                            linewidth=linewidths[tot_idx])
            elif plot_type_right == 'scatter':
                ax2_handle.scatter(x_vals, y_vals_right[y_idx_right][:, ue], 
                            alpha=0.6, color='b', 
                            linewidth=linewidths[tot_idx])
            elif plot_type_right == 'bar':
                ax2_handle.bar(x_vals, y_vals_right[y_idx_right][:, ue], 
                               alpha=0.6, color='b', 
                               linewidth=linewidths[tot_idx])
            else:
                raise Exception(f'No plot type named "{plot_type_right}" '
                                 'for right plot.')
        
        ax1_handle.set_title(f'UE {ue}')

        ax1_handle.set_xlabel(x_label)
        ax1_handle.set_ylabel(y_label[0], color='g')
        ax2_handle.set_ylabel(y_label[1], color='b')
        #ax1_handle.spines['right'].set_color('red')
        # Set tick colors as well
        # ax1_handle.tick_params(axis='y', colors='g')
        # ax2_handle.tick_params(axis='y', colors='b')
        tickspacing_ax1=tickspacing_ax2=np.zeros(len(ue_list))
        
        # Set number of tick information per plot.
        ue_idx = ue_list.index(ue)
        if limits_ax1:
            if len(ue_list) == 1 and limits_ax1 != []:
                # print(limits_ax1[0])
                # print(limits_ax1[1])
                ax1_handle.set_ylim(limits_ax1[0], limits_ax1[1])
                tickspacing_ax1 = ((limits_ax1[1] - limits_ax1[0]) / 
                                   no_ticks_ax1[ue_idx])
                ax1_handle.yaxis.set_major_locator(
                    ticker.MultipleLocator(base=tickspacing_ax1))
                
            elif len(ue_list) > 1 and limits_ax1[ue_idx] != []:
                ax1_handle.set_ylim(limits_ax1[ue_idx][0],
                                    limits_ax1[ue_idx][1])
                tickspacing_ax1[ue_idx] = \
                    ((limits_ax1[ue_idx][1] - limits_ax1[ue_idx][0]) / 
                     no_ticks_ax1[ue_idx])
                ax1_handle.yaxis.set_major_locator(
                    ticker.MultipleLocator(base=tickspacing_ax1[ue_idx]))
        
        if limits_ax2:
            if len(ue_list) == 1 and limits_ax2 != []:
               # print(limits_ax2[0])
               # print(limits_ax2[1])
               ax2_handle.set_ylim(limits_ax2[0], limits_ax2[1])
               tickspacing_ax2 = ((limits_ax2[1] - limits_ax2[0]) / 
                                   no_ticks_ax2[ue_idx])
               ax2_handle.yaxis.set_major_locator(
                   ticker.MultipleLocator(base=tickspacing_ax2))
            elif len(ue_list) > 1 and limits_ax2[ue_idx] != []:
                 ax2_handle.set_ylim(limits_ax2[ue_idx][0], 
                                     limits_ax2[ue_idx][1])
                 tickspacing_ax2[ue_idx] = \
                     ((limits_ax2[ue_idx][1] - limits_ax2[ue_idx][0]) / 
                      no_ticks_ax2[ue_idx])
                 ax2_handle.yaxis.set_major_locator(
                     ticker.MultipleLocator(base=tickspacing_ax2[ue_idx]))
        # for the ticks: (label 1 and label 2, there is a difference!)
        # for tick in ax2_handle.yaxis.get_major_ticks():
        #     tick.label2.set_fontsize(19) 
        
        # For the font of the axis label only    
        lab = ax1_handle.yaxis.get_label()
        lab.set_fontsize(label_fonts[0])            
        lab = ax2_handle.yaxis.get_label()
        lab.set_fontsize(label_fonts[1])
        
        axs1.append(ax1_handle)
        axs2.append(ax2_handle)
        
        if fill:
            if isinstance(fill_var, str):
                raise Exception('no var provided for filling')
            
            low_lim, high_lim = ax1_handle.get_ylim()
            
            ax1_handle.fill_between(x_vals, 0, high_lim, 
                                    where=fill_var[:,ue] > 0, 
                                    color=fill_color, 
                                    alpha=linewidths[-1],
                                    label=fill_label)
            
            if use_legend and legend_inside:
                    ax1_handle.legend(loc=legend_loc)
    
        
        # Make Limits for left and right axis, in case information is given 
        # for that particular plot 
        if limits_ax1:
            if len(ue_list) == 1 and limits_ax1 != []:
                ax1_handle.set_ylim(limits_ax1)
            elif len(ue_list) > 1 and limits_ax1[ue] != []:
                ax1_handle.set_ylim(limits_ax1[ue])
                
        
        if limits_ax2:
            if len(ue_list) == 1 and limits_ax2 != []:
                ax2_handle.set_ylim(limits_ax2)
            elif len(ue_list) > 1 and limits_ax2[ue] != []:
                ax2_handle.set_ylim(limits_ax2[ue])
        
    
    if use_legend and not legend_inside:
        handles, labels = ax1_handle.get_legend_handles_labels()
        legend_handle = fig.legend(handles, labels, loc="center", 
                                   bbox_to_anchor=legend_loc,
                                   fancybox=True, shadow=True, ncol=5)
        
    if title != '':
        fig.suptitle(title)
    
    
    if savefig:
        if filename == '':
            filename = ut.get_time()
        if use_legend and not legend_inside:
            fig.savefig(filename,
                        bbox_extra_artists=(legend_handle,), 
                        bbox_inches='tight')
        else:
            plt.savefig(filename)
        
        print(f'Saved: {filename}')
        
    plt.show()
 

def t_student_mapping(N, one_sided=True, confidence=0.95):
    # N samples
    if ut.parse_input_type(N, ['int']):
        Exception('N is the number of samples. '
                  'Needs to be an integer.')
    
    if not one_sided or confidence != 0.95:
        Exception('Not implemented.')
        # Ctrl + F for 'Table of selected' values in:
        # https://en.wikipedia.org/wiki/Student%27s_t-distribution
    
    r = N - 1
        
    t_95_map = {1:	6.314,
                2:	2.920,
                3:	2.353,
                4:	2.132,
                5:	2.015,
                6:	1.943,
                7:	1.895,
                8:	1.860,
                9:	1.833,
                10: 1.812,
                11: 1.796,
                12: 1.782,
                13: 1.771,
                14: 1.761,
                15: 1.753,
                16: 1.746,
                17: 1.740,
                18: 1.734,
                19: 1.729,
                20: 1.725,
                21: 1.721,
                22: 1.717,
                23: 1.714,
                24: 1.711,
                25: 1.708,
                26: 1.706,
                27: 1.703,
                28: 1.701,
                29: 1.699,
                30: 1.697,
                40: 1.684,
                50: 1.676,
                60: 1.671,
                80: 1.664,
                100: 1.660,
                120: 1.658,
                200: 1.645}
        
    if r <= 0:
        Exception('Please...')
    if r <= 30 or r in [40, 50, 60, 80, 100, 120, 200]:
        t_95 = t_95_map[r]
    else: # we'll do some interpolation...
        if r < 40:
            x1 = 30
            x2 = 40
        elif r < 50:
            x1 = 40
            x2 = 50
        elif r < 60:
            x1 = 50
            x2 = 60
        elif r < 80:
            x1 = 60
            x2 = 80
        elif r < 100:
            x1 = 80
            x2 = 100
        elif r < 120:
            x1 = 100
            x2 = 120
        elif r < 200:
            x1 = 120
            x2 = 200
        
        if r > 200:
            # 200 is used as infinite, the value converges to a limit
            t_95 = t_95_map[200]
        else:
            slope = (t_95_map[x2] - t_95_map[x1]) / (x2 - x1)
            delta = r - x1
            t_95 = t_95_map[x1] + slope * delta 
        
    return round(t_95, 4)


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


##############################################################################

def init_2D_data(n1, n2):
    data = ut.make_py_list(2, [n1, n2])
    for i in range(n1):
        for j in range(n2):
            data[i][j] = None
    return data


def init_sim_data(sim_data_loaded, sim_data_trimmed, sim_data_computed,
                  vars_load, vars_compute, ttis, set_of_files, 
                  ttis_temp, set_of_files_temp,
                  always_load, always_trim, always_compute):
    
    """
    [trace_idx, variable_idx]
    """
    
    ttis_changed = np.array_equal(ttis, ttis_temp)
    # in the future, replace this loop for the proper python function 
    # compare the eleements of two lists.
    files_changed = ut.elementwise_comparison_loop(set_of_files, 
                                                   set_of_files_temp)
    
    implementation_mode = (always_load or always_trim or always_compute)
    
    if not implementation_mode and not ttis_changed and not files_changed:
        # keep the same value they have, no change is needed
        pass
    else:
    
        n_vars_load = len(vars_load)
        n_vars_comp = len(vars_compute)
        n_files = len(set_of_files)
        
        # Initialize lists with None to state that operations need to happen
        
        if implementation_mode or ttis_changed or files_changed:
            sim_data_computed = init_2D_data(n_files, n_vars_comp)
        
        if always_trim or always_load or ttis_changed or files_changed:
            sim_data_trimmed = init_2D_data(n_files, n_vars_load)
        
        if always_load or files_changed:
            sim_data_loaded = init_2D_data(n_files, n_vars_load)
        
    return (sim_data_loaded, sim_data_trimmed, sim_data_computed)
        
   
def get_vars_to_load(idx, vars_to_load_names):
    """
    This function is just a dictionary with a list of variables that need
    to be loaded for each index. In order to return actual words, we convert
    those words to the actual variables from the list given as 2nd argument.
    """
    # What variables need to be loaded to garantee everything needed for the
    # computations is in memory?
    
    # Help regarding numbers:
    # 'sinr_diff':            'sinr_diff' -> [2,3]
    # 'running_avg_bitrate':  'bitrate realised' -> [4]
    # 'rolling_avg_bitrate':  'sp', 'bitrate_realise'' -> [4]
    # 'instantaneous_bler':   'block_errors', 'n_blocks' -> [5, 6]
    # 'running_avg_bler':     'block_errors', 'n_blocks' -> [5,6]
    # 'signal_power_db':      'experienced_signal_power' [10]
    # 'real_interference_db': 'real_dl_interference' [12]
    # 'est_interference_db':  'est_dl_interference' [13]
    # 'beam_formula':         'beams_used' [7]
    # 'beam_sum':             'beams_used' [7]
    # 'freq_vec':             'sp' [0]
    # 'frames':               'buffers', -> [1]
    # 'I_frames':             'buffers', -> [1]
    # 'avg_packet_lat':       'buffers', -> [1]
    # 'avg_packet_drop_rate': 'buffers -> [1]
    # 'count_ues_scheduled':  'scheduled_UEs' -> [15]
    # 'count_ues_bitrate':    'scheduled_UEs' -> [15]
    
    load_dict = {0.1: [16], 0.2: [17],
                 1: [4], 1.1: [4], 1.2: [0,4],
                 2: [2,3], 2.1: [2,4,5,6], 2.15: [2,4,5,6], 2.2: [2,4,8], 
                 2.3: [2,3], 2.4: [2,3,5,6],
                 3: [11], 3.1: [11], 3.2: [10], 3.3: [10,12], 3.4: [10,12], 
                 3.45: [10,12], 3.5: [10,12], 3.6: [10,12], 3.7: [12,13],
                 3.8: [12,13],
                 4.1: [9], 4.2: [9], 4.3: [4,9],
                 5.1: [7], 5.2: [7], 5.3: [7], 5.4: [2,5,6], 5.5: [5,6,7], 
                 7.1: [5,6], 7.2: [5,6], 7.3: [5,6],
                 7.35: [5,6], 7.4: [4,5,6], 7.5: [2,5,6],
                 9.1: [5,6,8], 9.2: [5,6,8], 9.3: [8,9], 9.4: [5,6,8],
                 10.1: [1], 10.15: [1], 10.2: [1], 10.25: [1], 
                 10.3: [1], 10.31: [1], 10.4: [1], 10.45: [1], 
                 10.5: [1], 10.55: [1], 10.6: [1], 10.65: [1], 
                 10.7: [], 10.8: [1], 10.9: [1],
                 11: [15], 11.1: [15], 11.2: [15], 11.3: [4], 
                 13: [14],
                 14.1: [1], 14.2: [1], 15: [7]
                 }
    
    # Always add 'sp' variable and return list of var names.
    return ['sp'] + [vars_to_load_names[i] for i in load_dict[idx]]


def get_vars_to_compute(idx, vars_to_compute_names):
    """
    This function is just a dictionary with a list of variables that need
    to be computed for each index. In order to return actual words, we convert
    those words to the actual variables from the list given as 2nd argument.
    """
    
    # What variables needs to be computed for this index?
    # Note: There may be variables that plot what has been loaded,
    #       and so they don't require further computations (hence, empty lists)
    
    # Another, more subtle, note: some rare times, it is worth for us to switch
    # the order with which we provide the computation indices if, for computing
    # e.g. avg_pck_lat_per_frame we very easily can compute avg_packet_lat as
    # well. So, actually, we include such computation in avg_pck_lat_per_frame
    # and we don't need to compute it again when it comes back.
    compute_dict = {0.1: [], 0.2: [],
                    1: [], 1.1: [1], 1.2: [2],
                    2: [], 2.1: [3], 2.15: [3], 2.2: [], 
                    2.3: [0], 2.4: [0,3],
                    3: [11], 3.1: [6,11], 3.2: [], 3.3: [], 3.4: [5,7], 
                    3.45: [5,7], 3.5: [5,7], 3.6: [5,7], 3.7: [],
                    3.8: [7,8],
                    4.1: [], 4.2: [], 4.3: [],
                    5.1: [9], 5.15: [26], 5.2: [], 
                    5.3: [10], 5.4: [10], 5.5: [3,10], 
                    7.1: [3], 7.2: [3,4], 7.3: [3,4],
                    7.35: [3,4], 7.4: [3], 7.5: [3],
                    9.1: [3], 9.2: [3], 9.3: [], 9.4: [3],
                    10.1: [12,14], 10.15: [12,14], 10.2: [12,15], 
                    10.25: [12,15], 10.3: [12,14,15], 10.31: [12,14,15], 
                    10.4: [12,13,14,15], 10.45: [12,13,14,15], 
                    10.5: [16], 10.55: [16], 10.6: [17], 10.65: [17], 
                    10.7: [14,15,16,17,18,19,20,21,22,23], 
                    10.8: [12,13,15], 10.85: [12,13,14], 
                    10.9: [12,13,18,19,20,21,14,15], 10.11: [],
                    11: [24], 11.1: [], 11.2: [], 11.3: [25], 
                    13: [],
                    14.1: [], 14.2: [], 15: []
                    }
            
    return [vars_to_compute_names[i] for i in compute_dict[idx]]
    
   
def load_sim_data(files, all_load_var_names, vars_to_load, sim_data_loaded):
    """
    Loads the variables from each file and prepares the data.
    Based on the index, only the required variables are loaded, if they 
    were not loaded before (i.e. if they are not empty).
    """
    
    n_files = len(files)
    
    # Load results from simulation (stats) files
    for f in range(n_files):
        file_path = files[f]
        for v in vars_to_load:
            v_idx = all_load_var_names.index(v)
            
            if sim_data_loaded[f][v_idx] is None:
                sim_data_loaded[f][v_idx] = ut.load_var_pickle(v, file_path)
        
        
def trim_sim_data(sim_data_loaded, sim_data_trimmed, all_load_var_names, 
                  vars_to_trim, files, trim_ttis):
    
    # Check if ttis to trim exists 
    sim_ttis = sim_data_loaded[0][0].sim_TTIs
    
    if not 0 <= trim_ttis[0] <= sim_ttis:
        raise Exception(f'TRIM_TTIS out of bounds! Sim TTIs = {sim_ttis}')
    
    # Note: always trim an empty variable and not all variables need triming!
    
    vars_to_not_trim = ['sp', 'buffers']
    
    # Within the variables to trim, which have per layer information that
    # require further trimming (e.g. to select one layer or single-layer mode):
    vars_with_no_layer = ['realised_bitrate_total', 'experienced_signal_power',
                          'olla', 'su_mimo_setting', 'channel', 
                          'scheduled_UEs']
    
    # Convert to NumPy arrays and trim to obtain only the useful parts
    for f in range(len(files)):
        for v in vars_to_trim:
            v_idx = all_load_var_names.index(v)
            
            if v in vars_to_not_trim:
                sim_data_trimmed[f][v_idx] = sim_data_loaded[f][v_idx]
                continue
            
            if sim_data_loaded[f][v_idx] is None and \
               sim_data_trimmed[f][v_idx] is None:
                raise Exception("Can't trim var not loaded.")
            
            if sim_data_loaded[f][v_idx] == []:
                print(f"Var '{v}' not computed during simulation.")
                continue
            
            # print(v)
            # TODO: take the numpy arrays from here and save it in the sim.py
            sim_data_trimmed[f][v_idx] = \
                np.array(sim_data_loaded[f][v_idx])[trim_ttis[0]:trim_ttis[1]]
            
            if v in vars_with_no_layer:
                continue
            
            # Select the layer we want (single-layer plot for now)
            
            l_idx = 0
            sim_data_trimmed[f][v_idx] = sim_data_trimmed[f][v_idx][:,:,l_idx]
            
            # # Select the downlink ttis only
            #sim_data[f][v_idx] = np.delete(sim_data[v_idx, f], ul_ttis, axis=0)
    
    # UL tti formula, if first and last tti are multiples of 5.
    # ul_ttis = np.arange(0, last_tti - first_tti, 5) - 1
    # ul_ttis = ul_ttis[1:]
    
    pass


def compute_sim_data(plot_idx, ues, ttis, 
                     all_loadable_var_names, all_computable_var_names, 
                     vars_to_compute, vars_to_trim,
                     sim_data_trimmed, sim_data_computed,
                     file_set):
    
    # Setup some useful variables:
    n_ues = len(ues)
    n_ttis = len(ttis)
    n_files = len(file_set)
    
    # Let's handle first the single_trace computations
    # In this first computation phase, we compute variables per trace only.
    for f in range(n_files):
        # Current file simulation parameters
        f_sp = sim_data_trimmed[f][0]
        GoP = f_sp.GoP
        
        # Count number of periods and frames
        n_periods = round(ttis[-1] *  f_sp.TTI_dur_in_secs * (GoP - 1))
        n_frames = n_periods * GoP
        
        for var_to_compute in vars_to_compute:
            # Index of where to put our freshly computed variable
            v = all_computable_var_names.index(var_to_compute)
            
            # Check trim dependencies, to make sure the variable has been 
            # trimmed properly. Otherwise, we can't continue with computation
            problems_with_trim_vars = False
            # find which variable failed the test to provide further info
            for v_trim in vars_to_trim:
                v_trim_idx = all_loadable_var_names.index(v_trim)
                if sim_data_trimmed[f][v_trim_idx] is None or \
                   sim_data_trimmed[f][v_trim_idx] == []:
                    print(f"Can't compute var '{var_to_compute}' because "
                          f"{v_trim} wasn't trimmed successfully")
                    problems_with_trim_vars = True
                    break
                
            if problems_with_trim_vars:
                break
            
            # COMPUTE INDEX 0: SINR difference
            if var_to_compute == 'sinr_diff' and \
               sim_data_computed[f][v] is None:
                # 'realised_SINR' is IDX 2 and 'estimated_SINR' is IDX 3
                sim_data_computed[f][v] = (sim_data_trimmed[f][2] - 
                                           sim_data_trimmed[f][3])
            
            # COMPUTE INDEX 1: Running bitrate 
            if var_to_compute == 'running_avg_bitrate' and \
               sim_data_computed[f][v] is None:
                # IDX 4 is for realised bit rate
                sim_data_computed[f][v] = \
                    np.zeros(sim_data_trimmed[f][4].shape)
                
                for ue in ues:
                    for tti in range(n_ttis):
                        sim_data_computed[f][v][tti,ue] = \
                            (sum(sim_data_trimmed[f][4][0:tti,ue]) / (tti+1))
                
                
            # COMPUTE INDEX 2: Rolling avg bitrate
            if var_to_compute == 'rolling_avg_bitrate' and \
               sim_data_computed[f][v] is None:
                # IDX 4 is for realised bit rate
                sim_data_computed[f][v] = \
                    np.zeros(sim_data_trimmed[f][4].shape)
                
                # TTIs of 1 GoP, interval over which to average the bit rate
                rolling_int = int(GoP / f_sp.FPS / f_sp.TTI_dur_in_secs)
                    
                for ue in ues:
                    for tti in range(n_ttis):
                        if tti < rolling_int:
                            sim_data_computed[f][v][tti,ue] = \
                                sum(sim_data_trimmed[f][4][0:tti,ue]) / (tti+1)
                        else:
                            sim_data_computed[f][v][tti,ue] = \
                                (sum(sim_data_trimmed[f][4]
                                     [tti-rolling_int:tti,ue]) / 
                                 rolling_int)
            
            
            # COMPUTE INDEX 3: Instantaneous BLER 
            if var_to_compute in ['instantaneous_bler',
                                  'running_avg_bler'] and \
               sim_data_computed[f][v] is None:
                # IDX 5 is blocks_with_errors
                sim_data_computed[f][v] = \
                    np.zeros(sim_data_trimmed[f][5].shape)
                # IDX 6 is n_transport_blocks
                for ue in ues:
                    sim_data_computed[f][v][:,ue] = \
                        np.array([sim_data_trimmed[f][5][i,ue] / 
                                  sim_data_trimmed[f][6][i,ue] * 100
                                  if sim_data_trimmed[f][6][i,ue] != 0 else 0 
                                  for i in range(n_ttis)])
            
                
            # COMPUTE INDEX 4: Running average BLER
            if var_to_compute == 'running_avg_bler' and \
               sim_data_computed[f][v] is None:
                # IDX 5 is block_errors
                sim_data_computed[f][v] = \
                    np.zeros(sim_data_trimmed[f][5].shape)
                # Requires instantaneous_bler, IDX 3 of computed
                for ue in ues:
                    for tti in range(n_ttis):
                        sim_data_computed[f][v][tti,ue] = \
                            (sim_data_computed[f][v][tti-1,ue] * tti + 
                             sim_data_computed[f][3][tti,ue]) / (tti+1)
                
            # COMPUTE INDEX 5: Signal Power in dBW
            if var_to_compute == 'signal_power_db' and \
               sim_data_computed[f][v] is None:
                # IDX 10 is the experienced signal power in Watt
                sim_data_computed[f][v] = (10 * 
                                           np.log10(sim_data_trimmed[f][10]))
            
            # COMPUTE INDEX 6: Signal Power per PRB in dBW
            if var_to_compute == 'signal_power_prb_db' and \
               sim_data_computed[f][v] is None:
                # IDX 11 is the experienced signal power in Watt
                sim_data_computed[f][v] = (10 * 
                                           np.log10(sim_data_trimmed[f][11]))
               
            # COMPUTE INDEX 7: Real Interference in dBW
            if var_to_compute == 'real_interference_db' and \
               sim_data_computed[f][v] is None:
                # IDX 12 is the experienced interference power in Watt
                sim_data_computed[f][v] = (10 * 
                                           np.log10(sim_data_trimmed[f][12]))
            
            # COMPUTE INDEX 8: Estimated Interference in dBW
            if var_to_compute == 'est_interference_db' and \
               sim_data_computed[f][v] is None:
                # IDX 13 is the estimated interference power in Watt
                sim_data_computed[f][v] = (10 * 
                                           np.log10(sim_data_trimmed[f][13]))
            
            # COMPUTE INDEX 9: Beam Formula Simple
            if var_to_compute == 'beam_formula_simple' and \
               sim_data_computed[f][v] is None:
                # IDX 7 is the beams_used
                sim_data_computed[f][v] = (sim_data_trimmed[f][7][:,:,0] + 
                                           sim_data_trimmed[f][7][:,:,1] * 10)
            
            
            # COMPUTE INDEX 10: Beam Sum
            if var_to_compute == 'beam_sum' and \
               sim_data_computed[f][v] is None:
                # IDX 7 is the beams_used
                sim_data_computed[f][v] = (sim_data_trimmed[f][7][:,:,0] + 
                                           sim_data_trimmed[f][7][:,:,1])
            
            # COMPUTE INDEX 11: Vector of Frequencies
            if var_to_compute == 'freq_vec' and \
               sim_data_computed[f][v] is None:
                if f_sp.n_prb > 1:
                    prb_bandwidth = f_sp.bandwidth / f_sp.n_prb
                    sim_data_computed[f][v] = \
                        (f_sp.freq - f_sp.bandwidth/2 + prb_bandwidth * 
                         np.arange(0, f_sp.n_prb * f_sp.freq_compression_ratio))
                else:
                    sim_data_computed[f][v] = [f_sp.freq]
            
            # COMPUTE INDEX 12 & 13: Vector of Frame Indices
            vars_requiring_12_or_13 = \
                ['frames', 'I_frames', 'avg_packet_lat', 'avg_packet_drop_rate',
                 'avg_pck_lat_per_I_frame', 'avg_pck_lat_per_P_frame', 
                 'avg_pck_drop_rate_per_I_frame', 
                 'avg_pck_drop_rate_per_P_frame',
                 'avg_pck_lat_per_frame_in_gop', 
                 'avg_pck_drop_rate_per_frame_in_gop']
            
            if var_to_compute in vars_requiring_12_or_13 and \
               sim_data_computed[f][v] is None:
                # Generate frame list
                sim_data_computed[f][12] = np.arange(n_frames)
                
                # Generate which of those are I frames:
                sim_data_computed[f][13] = np.zeros([n_frames, n_ues])
                sim_data_computed[f][13][0,:] = 1
                for frame_idx in range(n_frames):
                    if frame_idx % GoP == 0:
                        sim_data_computed[f][13][frame_idx,:] = 1
            
            # COMPUTE INDEX 14: Average Packet Latency 
            if var_to_compute in ['avg_packet_lat',
                                  'avg_pck_lat_per_frame'] and \
               sim_data_computed[f][v] is None:
                   
                sim_data_computed[f][v] = np.zeros([n_frames, n_ues])
                
                for ue in ues:
                    for per in range(n_periods):
                        # Loop over frame indices (fi)
                        for fi in range(GoP):
                            # IDX 1 are the buffers
                            f_info = \
                                sim_data_trimmed[f][1][ue].frame_infos[per][fi]
                            sim_data_computed[f][v][fi][ue] = \
                                (f_info.avg_lat.microseconds / 1e3)
                                
            # COMPUTE INDEX 15: Average Packet Drop Rate
            if var_to_compute in ['avg_packet_drop_rate',
                                  'avg_pck_drop_rate_per_frame'] and \
               sim_data_computed[f][v] is None:
                   
                sim_data_computed[f][v] = np.zeros([n_frames, n_ues])
                
                # Compute packet success percentages, average latencies and 
                # drop rates
                for ue in ues:
                    for per in range(n_periods):
                        for frm in range(GoP):
                            f_info = \
                                sim_data_trimmed[f][1][ue].frame_infos[per][frm]
                            packets_sent = f_info.successful_packets
                            dropped_packets = f_info.dropped_packets
                            total_packets = \
                                packets_sent + dropped_packets
                            
                            frame_idx = per * GoP + frm
                            
                            sim_data_computed[f][v][frame_idx][ue] = \
                                dropped_packets / total_packets * 100
            
            
            # COMPUTE INDEX 16: Average Packet Latency across all frames
            if var_to_compute == 'avg_pck_lat_per_frame' and \
               sim_data_computed[f][v] is None:
                
                # IDX 14 is the avg_packet_latency
                aux = np.zeros(sim_data_computed[f][14].shape)
                aux[:] = sim_data_computed[f][14][:] 
                
                # I frames have more packets than P frames, so we need to 
                # account for that when computing the overall average ... 
                # from per frame measurements.
                
                # Scale up I frames:  (IDX 13 is the I frame indices )
                aux[sim_data_computed[f][13], :] *= \
                    (1 / f_sp.IP_ratio)
                
                # Don't try to understand this math... Basically we needed to 
                # weigh the number of packets of each frame and divide by the 
                # number of frames.
                b = (1/f_sp.IP_ratio) * ((GoP - 1) * f_sp.IP_ratio + 1)
                
                # IDX 12 is the frame indices
                sim_data_computed[f][v] = \
                    np.round(np.sum(aux, 0) / 
                             (len(sim_data_computed[f][12]) * b / GoP), 2)
                                  
            
            # COMPUTE INDEX 17: Average Packet Drop Rate across all frames
            if var_to_compute == 'avg_pck_drop_rate_per_frame' and \
               sim_data_computed[f][v] is None:
                # IDX 15 is the avg_packet_drop_rate
                aux= np.zeros(sim_data_computed[f][15].shape)
                aux[:] = sim_data_computed[f][15][:] 
                
                # Scale up I frames: (IDX 13 is the I frame indices )
                aux[sim_data_computed[f][13], :] *= 1 / f_sp.IP_ratio
                
                # Don't try to understand this math... Basically we needed to 
                # weigh the number of packets of each frame and divide by the 
                # number of frames.
                b = (1 / f_sp.IP_ratio) * ((GoP - 1) * f_sp.IP_ratio + 1)
                
                sim_data_computed[f][v] = \
                    np.round(np.sum(aux, 0) / 
                             (len(sim_data_computed[f][12]) * b / GoP), 2)
                    
            # COMPUTE INDEX 18-23: 
            if var_to_compute in ['avg_pck_lat_per_I_frame',
                                  'avg_pck_lat_per_P_frame',
                                  'avg_pck_drop_rate_per_I_frame',
                                  'avg_pck_drop_rate_per_P_frame',
                                  'avg_pck_lat_per_frame_in_gop',
                                  'avg_pck_drop_rate_per_frame_in_gop'] and \
               sim_data_computed[f][v] is None:
                
                # METRICS PER FRAME IN THE GOP: lat and drop rate
                sim_data_computed[f][22] = np.zeros([GoP, n_ues])
                sim_data_computed[f][23] = np.zeros([GoP, n_ues])
                
                
                I_frame_idxs = np.array([i for i in range(n_frames) 
                                         if i % GoP == 0])
                
                # IDX 14 and 15 are avg_lat and drop_rate
                for idx_in_gop in range(GoP):
                    idxs = I_frame_idxs + idx_in_gop
                    sim_data_computed[f][22][idx_in_gop,:] = \
                        np.mean(sim_data_computed[f][14][idxs, :], 0)
                    sim_data_computed[f][23][idx_in_gop,:] = \
                        np.mean(sim_data_computed[f][15][idxs, :], 0)
                
                # PER I frame
                sim_data_computed[f][18] = sim_data_computed[f][22][0, :]
                sim_data_computed[f][20] = sim_data_computed[f][23][0, :]
                
                # PER P frame
                sim_data_computed[f][19] = \
                    sum(sim_data_computed[f][22][1:, :],0) / (GoP - 1)
                sim_data_computed[f][21] = \
                    sum(sim_data_computed[f][23][1:, :],0) / (GoP - 1)
                    
                    
                # PER FRAME METRICS
                # Adjust for proportionality on # pcks per frame
                P_frame_pcks_proportion = (GoP - 1) * f_sp.IP_ratio 
                # denominator
                d = P_frame_pcks_proportion + 1
                sim_data_computed[f][16] = \
                    (sim_data_computed[f][18] + (P_frame_pcks_proportion * 
                                                 sim_data_computed[f][19])) / d
                sim_data_computed[f][17] = \
                    (sim_data_computed[f][20] + (P_frame_pcks_proportion * 
                                                 sim_data_computed[f][21])) / d
                
                # Round for clarity
                sim_data_computed[f][16] = np.round(sim_data_computed[f][16],2)
                sim_data_computed[f][17] = np.round(sim_data_computed[f][17],2)
                sim_data_computed[f][18] = np.round(sim_data_computed[f][18],2)
                sim_data_computed[f][19] = np.round(sim_data_computed[f][19],2)
                sim_data_computed[f][20] = np.round(sim_data_computed[f][20],2)
                sim_data_computed[f][21] = np.round(sim_data_computed[f][21],2)
                sim_data_computed[f][22] = np.round(sim_data_computed[f][22],2)
                sim_data_computed[f][23] = np.round(sim_data_computed[f][23],2)
        
            # COMPUTE INDEX 24: Count_ues_scheduled
            if var_to_compute == 'count_ues_scheduled' and \
               sim_data_computed[f][v] is None:
                # IDX 15 is scheduled UEs
                sim_data_computed[f][v] = np.sum(sim_data_trimmed[f][15], 1)
                
                # make it 2D: #ttis x 1
                sim_data_computed[f][v] = np.reshape(sim_data_computed[f][v], 
                                                     (n_ttis, 1))
        
            # COMPUTE INDEX 25: Count ues with bitrate
            if var_to_compute == 'count_ues_bitrate' and \
               sim_data_computed[f][v] is None:
                # IDX 4 is the realised_bitrate
                ues_with_bitrate = np.nan_to_num(sim_data_trimmed[f][4] / 
                                                 sim_data_trimmed[f][4])
                sim_data_computed[f][v] = np.sum(ues_with_bitrate, 1)
        
            # COMPUTE INDEX 26: Beam Formula Processed
            if var_to_compute == 'beam_formula_processed' and \
               sim_data_computed[f][v] is None:
                # INCLUDES COMPUTATION FOR beam_formula_simple, compute index 9
                # IDX 7 is the beams_used
                sim_data_computed[f][9] = (sim_data_trimmed[f][7][:,:,0] + 
                                           sim_data_trimmed[f][7][:,:,1] * 10)
                
                # Copy the old beam_formula
                sim_data_computed[f][v] = sim_data_computed[f][9]
                
                # Process the formula to make it smoother
                
                # Keep the formula constant if the next value is 0
                for tti in range(n_ttis):
                    for ue in ues:
                        if sim_data_computed[f][v][tti, ue] == 0:
                            sim_data_computed[f][v][tti, ue] = \
                                sim_data_computed[f][v][tti-1, ue]
                    
                # address first zeros too
                for ue in ues:
                    if sim_data_computed[f][v][0, ue] == 0:
                        # find next non-zero value
                        # idx = ut.first_nonzero(beam_formula, invalid_val=-1):
                        idx = (sim_data_computed[f][v][:, ue] != 0).argmax()
                        for tti in range(0, idx):
                            sim_data_computed[f][v][tti, ue] = \
                                sim_data_computed[f][v][idx, ue]
    
            # COMPUTE INDEX 27: GoP indices
            if var_to_compute == 'gop_idxs' and \
               sim_data_computed[f][v] is None:    
                # Plots for frame in GoP
                sim_data_computed[f][v] = np.arange(0, GoP)
        
            # COMPUTE INDEX 28: Avg. SINR across the trace
            if var_to_compute == 'avg_sinr' and \
               sim_data_computed[f][v] is None:
                   pass
    
    # If there are indications of multitrace, check whether there are 
    # multitrace vars to compute, e.g. like the average SINR across all traces.
    if n_files > 1:
        f_sp = sim_data_trimmed[f][0]
        GoP = f_sp.GoP
        for var_to_compute in vars_to_compute:
            # COMPUTE INDEX 29: Avg. SINR of each trace, averaged over traces
            if var_to_compute == 'avg_sinr_multitrace' and \
               sim_data_computed[f][v] is None:
                pass


def plot_sim_data(plot_idx, file_set, ues, ttis, x_vals, sim_data_trimmed, 
                  sim_data_computed, results_filename, base_folder, save_fig, 
                  save_format='svg'):
    
    """
        THE MEANING OF EACH PLOT INDEX IS IN PLOTS_PHASE1.PY.
    """
    
    x_label_time = 'Time [s]'
    n_ues = len(ues)
    n_files = len(file_set)
    
    ########################### PART OF PLOTING ##############################
    for f in range(n_files):
        f_sp = sim_data_trimmed[f][0]
        
        curr_file = file_set[f]
        
        # File naming
        freq_save_suffix = '_' + str(round(f_sp.freq / 1e9,1)) + 'ghz'
        stats_folder = curr_file.split('\\')[-2]
        save_folder = base_folder + stats_folder + '\\'
        file_name = save_folder + 'IDX-' + str(plot_idx) + freq_save_suffix
        
        if save_format in ['png', 'svg', 'pdf']:
            file_name += '.' + save_format
        else:
            raise Exception('Unsupported save format.')
        
        if save_fig and not ut.isdir(save_folder):
            ut.makedirs(save_folder)
                    
        # Avg. channel across time (sum over the PRBs, and antenna elements)
        if plot_idx == 0.1:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][16]],
                         x_axis_label=x_label_time, y_axis_label='Power [dBW]',
                         filename=file_name, savefig=save_fig)
        
        # The two indicies below require the channel_per_prb variable
        # Channel across time for many PRBs
        if plot_idx == 0.2:
            pass
        
        # Channel across PRBs for one time (tti)
        if plot_idx == 0.3:
            pass
        
        
        # Instantaneous Realised Bitrate
        if plot_idx == 1:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][4]], 
                          x_label_time, 'Mbps', 'Realised bitrate')
        
        
        # Inst. vs Running avg bitrate 
        if plot_idx == 1.1:
                plot_for_ues(ues, x_vals, [sim_data_trimmed[f][4], 
                                           sim_data_computed[f][1]], 
                             x_label_time, 'Mbps', 'Realised bitrate', 
                             savefig=save_fig)
    
    
        # Inst. vs Rolling avg bitrate 
        if plot_idx == 1.2:
            # Make constant line
            marker_line = np.ones(sim_data_trimmed[f][4].shape) * 80
        
            if n_ues == 1 and f_sp.avg_bitrate_dl < 400:
                # (Single-user case - normal bitrate)
                plot_for_ues(ues, x_vals, [sim_data_trimmed[f][2], 
                                           sim_data_computed[f][2]], 
                             x_axis_label=x_label_time, 
                             y_axis_label='Bit rate [Mbps]', 
                             title='', linewidths=[0.8, 2], 
                             y_labels=['Instantaneous', 
                                       'Rolling avg.\nover GoP duration'], 
                             use_legend=True, legend_inside=True, 
                             legend_loc=(0.64,0.2), ncols=1, size=1, 
                             filename=file_name, savefig=save_fig, 
                             same_axs=True) 
            
            if n_ues == 1 and f_sp.avg_bitrate_dl > 400:
                # (Single-user case - full buffer)
                plot_for_ues(ues, x_vals, 
                             [sim_data_trimmed[f][4], sim_data_computed[f][2]], 
                             x_axis_label=x_label_time, 
                             y_axis_label='Bit rate [Mbps]', 
                             title='', linewidths=[0.8, 2], 
                             y_labels=['Instantaneous', 
                                       'Rolling avg.\nover GoP duration'], 
                             use_legend=True, legend_inside=True, 
                             legend_loc=(0.645,.2), 
                             size=1, width=6.4, height=4, 
                             filename=file_name,
                             savefig=save_fig, same_axs=True) 
            
            if n_ues > 1:
                # (Multi-user case)
                plot_for_ues(ues, x_vals, [sim_data_trimmed[f][4], 
                                           sim_data_computed[f][2], 
                                           marker_line], 
                             x_axis_label=x_label_time, 
                             y_axis_label='Bit rate [Mbps]', 
                             title='', linewidths=[0.3, 1.5, 1], 
                             y_labels=['Instantaneous', 
                                       'Rolling avg. over GoP duration', 
                                       'Application Bit rate'], 
                             use_legend=True, ncols=3,
                             size=1.3, filename=file_name, 
                             savefig=save_fig, uniform_scale=[-8, 240]) 
            
                
        # SINR (Multi-user case)
        if plot_idx == 2:
            plot_for_ues(ues, x_vals, 
                         [sim_data_trimmed[f][3], sim_data_trimmed[f][2]], 
                         x_axis_label=x_label_time, y_axis_label='SINR [dB]', 
                         y_labels=['Estimated', 'Experienced'], 
                         use_legend=True, ncols=2, size=1.3,filename=file_name, 
                         savefig=save_fig)
    
        # SINR vs BLER: only when there are active transmissions (single_user)
        if plot_idx == 2.1:
            plot_for_ues_double([1], x_vals, 
                                [sim_data_trimmed[f][2]],
                                [sim_data_computed[f][3]], 
                                x_label_time, 
                                ['Experienced SINR [dB]', 'BLER [%]'],
                                linewidths=[1,0.4,0.15], 
                                limits_ax1=[15,22], no_ticks_ax1=[5],
                                label_fonts=[17,17], fill=True, 
                                fill_var=sim_data_trimmed[f][4], 
                                use_legend=True,
                                legend_loc=(1.02,.955), 
                                legend_inside=False,
                                fill_label='Active\ntransmissions',
                                width=7.8, height=4.8, size=1.2,
                                filename=file_name, 
                                savefig=save_fig)
        
        # SINR vs BLER: with active transmissions (multi-user)
        if plot_idx == 2.15:
            plot_for_ues_double(ues, x_vals, 
                                [sim_data_trimmed[f][2]],
                                [sim_data_computed[f][3]], 
                                x_label_time, 
                                ['Experienced SINR [dB]', 'BLER [%]'],
                                linewidths=[1,0.4,0.15], 
                                limits_ax1=[[15,22]] * n_ues, 
                                no_ticks_ax1=[5]*n_ues,
                                fill=True, fill_var=sim_data_trimmed[f][4], 
                                fill_label='Active\ntransmissions',
                                filename=file_name, 
                                savefig=save_fig)
            
            
         # SINR vs OLLA: with active transmissions (multi-user)
        if plot_idx == 2.2:
            plot_for_ues_double(ues, x_vals, 
                                [sim_data_trimmed[f][2]], 
                                [sim_data_trimmed[f][3]], x_label_time, 
                                ['Experienced SINR [dB]', '$\Delta_{OLLA}$'],
                                linewidths=[1,1,0.15], 
                                label_fonts=[13,16], fill=True, 
                                fill_var=sim_data_trimmed[f][4], 
                                use_legend=True,
                                legend_loc=(1.02,.955), 
                                legend_inside=False,
                                fill_label='Active\ntransmissions',
                                width=7.8, height=4.8, size=1.2,
                                filename=file_name, 
                                savefig=save_fig)
    
        if plot_idx == 2.3:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][0]], 
                         x_label_time, '[dB]', 
                         'SINR diff', filename=file_name, 
                         savefig=save_fig)
            
        
        if plot_idx == 2.4:
            plot_for_ues_double(ues, x_vals, [sim_data_computed[f][0]], 
                                [sim_data_computed[f][3]],
                                x_label_time, ['SINR diff [dB]', 'BLER [%]'],
                                filename=file_name, 
                                savefig=save_fig)
            
            
        # Signal power variation across PRBs
        if plot_idx == 3:
            if not f_sp.save_per_prb_variables:
                print("Cannot plot if not computed during SIM!")
                continue
            
            # antenna index: 
            tti_idx = 3
            
            if len(sim_data_computed[f][11]) > 1:
                plt_type = 'line'
            else:
                plt_type = 'scatter'
                
            plot_for_ues(ues, sim_data_computed[f][11], 
                         [sim_data_trimmed[f][11][tti_idx,:,:].T], 
                         'Frequency', 'Watt', 
                         'Signal power variation across frequency',
                         savefig=save_fig, plot_type=plt_type)
                
        
        # Signal power variation across PRBs in dB
        if plot_idx == 3.1:
            if not f_sp.save_per_prb_variables:
                print('Per PRB variables not saved during SIM!')
                continue
            
            # antenna index: 
            tti_idx = 3
            
            plot_for_ues(ues, sim_data_computed[f][11], 
                         [sim_data_computed[f][6]], 
                         'Frequency', 'dB', 
                         'Signal power variation across frequency',
                         savefig=save_fig)
            
        # Signal power 
        if plot_idx == 3.2:
            # Plot signal power variation across time
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][10]], 
                          x_label_time, 'Watt', 'signal power across time')
    
        # Signal power vs interference         
        if plot_idx == 3.3:        
            plot_for_ues(ues, x_vals, 
                         [sim_data_trimmed[f][10], sim_data_trimmed[f][12]], 
                          x_label_time, '[W]', 'Signal Power vs Interference')
             
        # Signal power vs interference (dBw) [double axis]
        if plot_idx == 3.4:
            plot_for_ues_double(ues, x_vals, [sim_data_computed[f][5]], 
                                [sim_data_computed[f][7]], x_label_time, 
                                ['Sig. Power [dBw]', 'Int. Power [dBw]'])
        
        # Signal power vs interference (dBw)  [single axis]
        if plot_idx == 3.45:
            plot_for_ues(ues, x_vals, 
                         [sim_data_computed[f][5], sim_data_computed[f][7]], 
                         x_label_time, y_axis_label='[dBw]', 
                         y_labels=['Signal', 'Interference'], 
                         filename=file_name, 
                         savefig=save_fig)
                
        # Signal power vs interference (dBm) [single]
        if plot_idx == 3.5:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][5] + 30, 
                                       sim_data_computed[f][7] + 30], 
                         x_label_time, 'Power [dBW]',
                         y_labels=['Signal', 'Interference'], use_legend=True,
                         ncols=2, size=1.3, width=6.4, height=4, 
                         filename=file_name, savefig=save_fig)
            
        # Signal power vs interference (dBm) [Double]
        if plot_idx == 3.6:
            plot_for_ues_double(ues, x_vals, 
                                [sim_data_computed[f][5] + 30], 
                                [sim_data_computed[f][7] + 30], 
                                x_label_time, 
                                ['Signal Power [W]', 'Interference Power [W]'])
    
        
        # Estimated vs Realised interference
        if plot_idx == 3.7:
            plot_for_ues(ues, x_vals, 
                         [sim_data_trimmed[f][13], sim_data_trimmed[f][12]], 
                          x_label_time, '[W]', 
                          'Estimated vs real interference')
         
        # Estimated vs Realised interference [dB]
        if plot_idx == 3.8:
            plot_for_ues(ues, x_vals, 
                         [sim_data_trimmed[f][13], sim_data_trimmed[f][12]], 
                          x_label_time, '[W]', 
                          'Estimated vs real interference')
                            
        # MCS same axs
        if plot_idx == 4.1:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][9]], x_label_time, 
                         'sim_data_trimmed[f][9] index',  
                         use_legend=True, legend_inside=True, 
                         legend_loc="lower right",
                         ncols=1, size=1.3, filename=file_name, 
                         savefig=save_fig, 
                         uniform_scale = [0.5, 15.5], same_axs=True)
            
        # MCS diff axs
        if plot_idx == 4.2:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][9]], x_label_time, 
                         'sim_data_trimmed[f][9] index', 
                         linewidths=[.4,.4,.4,.4], 
                         y_labels=['UE 0','UE 1','UE 2','UE 3'], 
                         ncols=1, size=1.3, filename=file_name, 
                         savefig=save_fig, uniform_scale = [6.5, 15.5], 
                         same_axs=False)
    
        # MCS and instantaneous bitrate per UE
        if plot_idx == 4.3:
            plot_for_ues_double(ues, x_vals, [sim_data_trimmed[f][9]], 
                                [sim_data_trimmed[f][4]], 
                                x_label_time, 
                                y_label=['sim_data_trimmed[f][9] index', 
                                         'Bit rate [Mbps]'],
                                savefig=save_fig)
        
        # Beams: best beam per user (filtered: prevents going back to zero when
        #        the UE is no scheduled. One plot per UE    
        if plot_idx == 5.1:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][9]], 
                         title='Formula: azi + ele x 10')
        
        # Beam formula processed (filtered and smoother) !
        if plot_idx == 5.15:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][26]], 
                         title='Formula: azi + ele x 10')
            
        # Beams filtered: doublePlot per UE for azi and elevation values.
        if plot_idx == 5.2:
            plot_for_ues_double(ues, x_vals, [sim_data_trimmed[f][7][:,:,0]], 
                                [sim_data_trimmed[f][7][:,:,1]],
                                x_label_time, 
                                y_label=['Azimuth [º]', 'Elevation[º]'], 
                                savefig=save_fig)
    
        # Beam sum: used to notice beam switching easily
        if plot_idx == 5.3:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][10]],
                         x_label_time, 'Azimuth + Elevation [º]', 
                         savefig=save_fig)
        
        # Beam sum: used to notice beam switching easily vs SINR
        if plot_idx == 5.4:
            plot_for_ues_double(ues, x_vals, [sim_data_computed[f][10]], 
                                [sim_data_trimmed[f][2]], 
                                x_label_time,
                                ['Azi. + El. [º]', 'SINR [dB]'],
                                savefig=save_fig)
        
        # Beam sum vs BLER
        if plot_idx == 5.5:
            plot_for_ues_double(ues, x_vals, [sim_data_computed[f][10]], 
                                [sim_data_computed[f][3]], x_label_time,
                                ['Azi. + El. [º]', 'BLER [%]'],
                                savefig=save_fig)
        
        # BLER: Instantaneous
        if plot_idx == 7.1:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][3]],
                         x_label_time, '%', '% of Blocks with errors',
                         savefig=save_fig)
        
        # BLER: Running Average
        if plot_idx == 7.2:       
            plot_for_ues(ues, x_vals, [sim_data_computed[f][4]], 
                         x_label_time, 'Avg. BLER [%]',
                         width=6.4, height=4.8, size=1.3, 
                         filename=file_name, 
                         same_axs=False, savefig=save_fig)
        
        # BLER: Instantaneous + Running Average
        if plot_idx == 7.3:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][3], 
                                       sim_data_computed[f][4]],
                         x_label_time, '%', 'Running average of BLER')
    
        if plot_idx == 7.35:
            plot_for_ues_double(ues, x_vals, [sim_data_computed[f][3]], 
                                [sim_data_computed[f][4]],
                                x_label_time, 
                                ['Inst. BLER [%]', 'Run. Avg. BLER [%]'], 
                                savefig=save_fig)
            
                    
            
        # BLER: instantaneous BLER and realised bitrate
        if plot_idx == 7.4:
            plot_for_ues_double(ues, x_vals, [sim_data_trimmed[f][4]],
                                [sim_data_computed[f][3]],
                                x_label_time, 
                                ['Inst. bitrate [Mbps]', 'BLER [%]'],
                                filename=file_name,
                                savefig=save_fig)
            
            
        # BLER: Instantaneous vs realised SINR
        if plot_idx == 7.5:
            plot_for_ues_double(ues, x_vals, [sim_data_computed[f][3]], 
                                [sim_data_trimmed[f][2]], 
                                x_label_time,
                                ['BLER [%]', 'SINR [dB]'],
                                savefig=save_fig)
            
        # OLLA (single-user) with Bitrate for grey areas
        if plot_idx == 9.1:
            # ONE UE ONLY:
            plot_for_ues_double([2], x_vals, [sim_data_computed[f][3]],
                                [sim_data_trimmed[f][8]], 
                                x_label_time, 
                                ['BLER [%]', '$\Delta_{OLLA}$'],
                                linewidths=[0.2,1,0.15], 
                                label_fonts=[13,16], fill=True, 
                                fill_var=sim_data_trimmed[f][2], 
                                use_legend=True,
                                legend_loc=(1.02,.955), 
                                legend_inside=False,
                                fill_label='Active\ntransmissions',
                                width=7.8, height=4.8, size=1.2,
                                filename=file_name, 
                                savefig=save_fig)
            
        # OLLA (multi-user) with active transmission (bitrate > 0)
        if plot_idx == 9.2:
            # MULTIPLE UEs
            plot_for_ues_double(ues, x_vals, [sim_data_computed[f][3]],
                                [sim_data_trimmed[f][8]], 
                                x_label_time, 
                                ['BLER [%]', '$\Delta_{OLLA}$'],
                                linewidths=[0.2,1,0.15], fill=True, 
                                fill_var=sim_data_trimmed[f][2],
                                fill_label='Active\ntransmissions',
                                width=7.8, height=4.8, size=1.2,
                                filename=file_name, 
                                savefig=save_fig)
        
        # OLLA: MCS vs olla
        if plot_idx == 9.3:
            plot_for_ues_double(ues, x_vals, [sim_data_trimmed[f][9]], 
                                [sim_data_trimmed[f][8]], x_label_time, 
                                ['CQI IDX', '$\Delta_{OLLA}$'], 
                                'sim_data_trimmed[f][9] and OLLA', 
                                [0.2,0.9])
         
        # OLLA: inst. bler vs olla        
        if plot_idx == 9.4:
            plot_for_ues_double([0, 2], x_vals, [sim_data_computed[f][3]], 
                                [sim_data_trimmed[f][8]], 
                                x_label_time, ['BLER [%]', '$\Delta_{OLLA}$'], 
                                'sim_data_trimmed[f][9] and OLLA', [0.2,0.9])
        
       
        
        # LATENCY and DROP RATE
        # avg latency across frames 
        if plot_idx == 10.1:
            plot_for_ues(ues, sim_data_computed[f][12], 
                         [sim_data_computed[f][14]],
                         'Frame index', 
                         'Avg. latency [ms]', '', [0.7,0.6],
                         filename=file_name, 
                         savefig=save_fig)
        
        # avg latency across frames (bar chart)
        if plot_idx == 10.15:
            plot_for_ues(ues, sim_data_computed[f][12], 
                         [sim_data_computed[f][14]],
                         'Frame index', 
                         'Avg. latency [ms]', '', [0.7,0.6],
                         filename=file_name, 
                         savefig=save_fig, plot_type='bar')
        
        # avg drop rate across frames
        if plot_idx == 10.2:
            plot_for_ues(ues, sim_data_computed[f][12], 
                         [sim_data_computed[f][15]],  
                         'Frame index', 
                         'Drop rate [%]', '', [0.7,0.6],
                         filename=file_name, 
                         savefig=save_fig)
        
        # avg drop rate across frames (bar chart)
        if plot_idx == 10.25:
            plot_for_ues(ues, sim_data_computed[f][12], 
                         [sim_data_computed[f][15]],  
                         'Frame index', 
                         'Drop rate [%]', '', [0.7,0.6],
                         filename=file_name, 
                         savefig=save_fig, plot_type='bar')
        
        # avg latency vs drop rate across frames (no I vs P distinction)
        if plot_idx == 10.3:    
            plot_for_ues_double(ues, sim_data_computed[f][12], 
                                [sim_data_computed[f][14]], 
                                [sim_data_computed[f][15]], 
                                'Frame index', 
                                ['Avg. latency [ms]', 'Drop rate [%]'], '', 
                                [0.7,0.6], filename=file_name, 
                                savefig=save_fig)
        
        # Same as 10.3 but showing the ticks and limits.
        if plot_idx == 10.31:    
            plot_for_ues_double(ues, sim_data_computed[f][12], 
                                [sim_data_computed[f][14]], 
                                [sim_data_computed[f][15]], 
                                'Frame index', 
                                ['Avg. latency [ms]', 'Drop rate [%]'], '', 
                                [0.7,0.6], filename=file_name, 
                                savefig=save_fig,
                                limits_ax1=[[0,5],[0,0.6],[0,5],[0,0.6]],
                                limits_ax2=[[-0.05,0.05],[-0.05,0.05],
                                            [-0.05,0.05],[-0.05,0.05]],
                                no_ticks_ax1=[4,4,4,4],no_ticks_ax2=[4,4,4,4])
        
        # Average latency and drop rate with I frame markings: line
        if plot_idx == 10.4:
            plot_for_ues_double(ues, sim_data_computed[f][12], 
                                [sim_data_computed[f][14]],
                                [sim_data_computed[f][15]], 'Frame index', 
                                ['Avg. latency [ms]', 'Drop rate [%]'],
                                linewidths=[0.6,.6,0.4], 
                                label_fonts=[14,14], fill=True, 
                                fill_var=sim_data_computed[f][13], 
                                fill_color='red',
                                use_legend=True, legend_loc=(.5,.0), 
                                legend_inside=False,
                                fill_label='I frame',
                                width=7.8, height=4.8, size=1.2,
                                filename=file_name,
                                savefig=save_fig, plot_type_left='line', 
                                plot_type_right='line')
            
        
        # Average latency and drop rate with I frame markings: bar
        if plot_idx == 10.45:
            plot_for_ues_double(ues, sim_data_computed[f][12], 
                                [sim_data_computed[f][14]],
                                [sim_data_computed[f][15]], 'Frame index', 
                                ['Avg. latency [ms]', 'Drop rate [%]'],
                                linewidths=[0.6,.6,0.4], 
                                label_fonts=[14,14], fill=True, 
                                fill_var=sim_data_computed[f][13], 
                                fill_color='red',
                                use_legend=True, legend_loc=(.5,.0), 
                                legend_inside=False,
                                fill_label='I frame',
                                width=7.8, height=4.8, size=1.2,
                                filename=file_name,
                                savefig=save_fig, plot_type_left='bar', 
                                plot_type_right='bar')
    
    
        # Prints average latency across frames
        if plot_idx == 10.5:
            # IDX 16 is average latency (one for each frame)
            avg_lat = round(np.mean(sim_data_computed[f][16]),2)
            avg_lat_std = round(np.std(sim_data_computed[f][16]),2)
            
            print(f'Done for folder: {stats_folder}\n' +
                  f'Avg: {avg_lat}\n'+
                  'Std: {avg_lat_std}')
            
                
        # Writes average latency across frames to file
        if plot_idx == 10.55:
            # IDX 16 is average latency (one for each frame)
            avg_lat = round(np.mean(sim_data_computed[f][16]),2)
            avg_lat_std = round(np.std(sim_data_computed[f][16]),2)
            
            s = f'{avg_lat}'
            s_std = f'{avg_lat_std}'
            
            print('Done for folder: ' + stats_folder + '. Result: ' + s)
            
            # append to a file the var above!
            with open(results_filename + '.csv', "a") as myfile:
                myfile.write(s + '\n')
            
            with open(results_filename + '_std.csv', "a") as myfile:
                myfile.write(s_std + '\n')
                
            # Also write the folder order
            with open(results_filename + '_folders.csv', "a") as myfile:
                myfile.write(stats_folder + '\n')
        
        # Prints average packet drop rate across frames
        if plot_idx == 10.6:
            # IDX 17 is average packet drop rate (one for each frame)
            avg_pdr = round(np.mean(sim_data_computed[f][17]),2)
            avg_pdr_std = round(np.std(sim_data_computed[f][17]),2)
            
            print(f'Done for folder: {stats_folder}\n' +
                  f'Avg: {avg_pdr}\n'+
                  'Std: {avg_pdr_std}')
            
        # Write average packet drop rate across frames to file
        if plot_idx == 10.65:
            # IDX 17 is average packet drop rate (one for each frame)
            avg_pdr = round(np.mean(sim_data_computed[f][17]),2)
            avg_pdr_std = round(np.std(sim_data_computed[f][17]),2)
            
            s = f'{avg_pdr}'
            s_std = f'{avg_pdr_std}'
            
            print('Done for folder: ' + stats_folder + '. Result: ' + s)
            
            # append to a file the var above!
            with open(results_filename + '.csv', "a") as myfile:
                myfile.write(s + '\n')
            
            with open(results_filename + '_std.csv', "a") as myfile:
                myfile.write(s_std + '\n')
                
            # Also write the folder order
            with open(results_filename + '_folders.csv', "a") as myfile:
                myfile.write(stats_folder + '\n')   
                    
        # prints all detailed measurements on frame information
        if plot_idx == 10.7:
            
            print(f'Analysis of folder {stats_folder}.')
            
            # Latency stats
            avg_latency = round(np.mean(sim_data_computed[f][16]),2)
            avg_latency_std = round(np.std(sim_data_computed[f][16]),2)     
            print(f'Avg. Latency is {avg_latency} ms,' + \
                  f' with STD of {avg_latency_std} ms.')
                
            print(f'Avg. latency per frames: {sim_data_computed[f][16]} ms.')
            print(f'Avg. latency for I frames: {sim_data_computed[f][18]} ms.')
            print(f'Avg. latency for P frames: {sim_data_computed[f][19]} ms.')
            
            # Drop rate stats
            avg_pdr = round(np.mean(sim_data_computed[f][17]),2)
            avg_pdr_std = round(np.std(sim_data_computed[f][17]),2)
            print(f'Avg. PDR is {avg_pdr} %,' + \
                  f' with STD of {avg_pdr_std} %.')
            
            print(f'Avg. drop rate per frames: '
                  f'{sim_data_computed[f][17]} %.')
            print(f'Avg. drop rate for I frames: '
                  f'{sim_data_computed[f][20]} %.')
            print(f'Avg. drop rate for P frames: '
                  f'{sim_data_computed[f][21]} %.')
    
        
        # Plots avg_pck_latency per frame of the GoP
        if plot_idx == 10.8:
            plot_for_ues(ues, sim_data_computed[f][27], 
                         [sim_data_computed[f][22]], '', 'ms', 
                          'Avg. Latency per frame in GoP', plot_type='bar')
        
        # Plots avg_pck_drop_rate per frame of the GoP
        if plot_idx == 10.9:    
            plot_for_ues(ues, sim_data_computed[f][27], 
                         [sim_data_computed[f][23]], '', '%', 
                          'Avg. drop rate per frame in GoP', plot_type='bar')
        
        # Did you know 10.1 == 10.10? So we need to jump over 10.10.
        
        # Plots the avg latency and pdr per frame of the GoP (double plot)
        if plot_idx == 10.11:
            print('before the mistake')
            plot_for_ues_double(ues, sim_data_computed[f][27], 
                                [sim_data_computed[f][22]],
                                [sim_data_computed[f][23]],
                                '', ['Latency [ms]', 'Drop Rate [%]'], 
                                'Avg. latency and drop rate per frame in GoP', 
                                plot_type_left='bar', plot_type_right='bar')
        
        # Scheduled UEs
        # Scheduled UEs: sum of co-scheduled UEs across time
        if plot_idx == 11:
            plot_for_ues([0], x_vals, [sim_data_computed[f][24]], 
                         same_axs=True)
            
            
        
        # Scheduled UEs: each UE is 1 when it is scheduled and 0 when it is not 
        if plot_idx == 11.1:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][15]], 
                         filename=file_name)
                    
            
        if plot_idx == 11.2:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][15]], 
                         filename=file_name, same_axs=True)
        
        
        # Count concurrent users by the bitrate
        if plot_idx == 11.3:
            plt.plot(x_vals, sim_data_computed[f][25])
            plt.xlabel(x_label_time)
            plt.ylabel('# co-scheduled UEs')
            
        # Number of co-scheduled layers per UE
        if plot_idx == 13:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][14]],
                         x_label_time, '# layers', 'Number of layers per UE')
    
        # Packet sequences all in the same plot
        if plot_idx == 14.1:
            for ue in ues:
                pck_seq = sim_data_trimmed[f][1][ue].parent_packet_seq
                pck_seq.plot_sequence(light=True, alpha=0.6)
            plt.show()
            
            if save_fig:
                plt.savefig('packet_sequences_same_axes.pdf', format='pdf')
        
            
        # Packet sequences all in different plots
        if plot_idx == 14.2:
            n_ue = n_ues
            if n_ue > 1:
                div_list = ut.divisors(n_ue)
                n1 = div_list[1]
                n2 = div_list[-2]
            else:
                n1 = 1
                n2 = 1
            
            width = 4.8
            height = 4
            size = 3
            r = width/height
            fig, axs = plt.subplots(n1,n2, tight_layout=True, 
                                    figsize=(r*height*size, size/r*width))
        
            if not isinstance(axs, np.ndarray):
                axs = [axs]
                
                
            for ue in ues:
            
                if n2 > 1:
                    aux = int(n_ues / 2)
                    if ue < aux:
                        idx = (0, ue)
                    else:
                        idx = (1, ue - aux)
                else:
                    idx = ues.index(ue)
                        
                ax_handle = axs[idx]
            
                ax_handle.plot()
                plt.sca(ax_handle)
                pck_seq = sim_data_trimmed[f][1][ue].parent_packet_seq
                pck_seq.plot_sequence(light=True, alpha=1)
                
                ax_handle.set_title(f'UE {ue}')
                ax_handle.set_xlabel(x_label_time)
                #ax_handle.set_ylabel('Packets per ms')
                
            
            fig.suptitle('Packets per ms')
            plt.show()
            if save_fig:
                plt.savefig('packet_sequences.pdf', format='pdf')
        
        
        if plot_idx == 15:
            
            bs_h = 3
            ue_h = 1.6
            h = 3 - ue_h
            
            # IDX 7 is beams
            x = h * np.tan(np.deg2rad(sim_data_trimmed[f][7][:,:,1]))
            y = h * np.tan(np.deg2rad(sim_data_trimmed[f][7][:,:,0]))
            plt.scatter(x,y)








