# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 09:51:23 2021

@author: Morais
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 14:34:43 2021

@author: Morais
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import csv

import utils as ut

try:
    # For MP4 short clips:
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage
except ModuleNotFoundError:
    # Error handling
    print('Could not find moviepy module. '
          'Did you pip it into the current env?'
          'Making GIFs and Videos will not work.')

# Note: there's probably a better way that avoids using the import below,
# see app_trafic_plots done in v2.
# import matplotlib.ticker as ticker 

def plot_for_ues(ue_list, x_vals, y_vals_left, y_vals_right=[],
                 x_axis_label='', 
                 y_axis_label=[''],
                 title='', linewidths='', tune_opacity=False, opacity=[],
                 y_labels_left='', y_labels_right='', y_axis_fonts=[], 
                 axes_tick_font_size=13,
                 xlim=[], ylim_left=[], ylim_right=[],
                 no_ticks_left=[], no_ticks_right=[],
                 tight_x_axis=True, tight_y_axis=True, 
                 fill=False, fill_var='', fill_color='grey', fill_label='',
                 use_legend=False, legend_inside=False, legend_loc="center",
                 ncols=1, size=1, width=30.0, 
                 height=10.0, same_axs=False, n1=-1, n2=-1,
                 plot_type_left='line', plot_type_right='line', 
                 savefig=False, filename='', saveformat='pdf'):
    """
    Parameters
    ----------
    ##### MANDATORY #####
    ue_list: UEs is the # of plots.
    
    x_vals: numpy.ndarray.
            x_vals are the values for the x axis of all plots.
            Should be [x_vals_of_ue, # UEs]. 
            But, since the x-axis is often the same, it can be 1-D and the same
            is used for all users.
    
    ############################33 CORRECT THESE FOR Y_VALS_LEFT AND RIGHT!!!!!!!!!!
    y_vals: numpy.ndarray.
            are the values for the y axis (right and/or left) of all plots
            Also, multiple data_points that share a common x-vals can be added.
            Each y-axis information, should be [y_vals_of_ue, # UEs]. 
            And, the comple y_axis should be a list of y-axis information, for 
            the amount of data_types we want to plot per plot.
            y[data_idx][:,ue] = y values of data data_idx for UE ue.
      
            The first y_vals dimension should agree with that of x_vals.
    
    ##### OPTIONAL #####

    use_legend: True if some texts should appear as legend
    legend_indide: True when the legend box is supposed to be inside
                     the plot
    legend_loc: the location. Can be text (or a code, see below), 
                  or tuple with (x,y) coordinates (from 0 to 1)
    
                    Location        String Location Code
                    'best'          0
                    'upper right'   1
                    'upper left'    2
                    'lower left'    3
                    'lower right'   4
                    'right'         5
                    'center left'   6
                    'center right'  7
                    'lower center'  8
                    'upper center'  9
                    'center'        10
                    
    ncols: columns of the legend.
    
    size, width and height are used together in the following formula:
        (fig height, fig width) = (r*height*size, size/r*width)
    
    filename
    savefig option

    same_axs: puts all ues in the same axis. 
              Cancels the subtitle on top of the subplot.
              Cancels the opacity and uses linewidths. -------------------> wuut?
    
    plot_type_left/right: plot type of the left/right y-axis
                          y_vals must have values for that axis, otherwise it
                          is not used.
                          Available options: 'line', 'scatter', 'bar'
      
    ylims: list of tuples, one for each UE. If there's just one tuple,
             it is used for all UEs
    
    n1 & n2: Option a) n1 * n2 must be = # UEs and that will be the
                         number of rows and columns of the subplots.
               Option b) (n1 = -1 and n2 != -1) or (n1 != -1 and n2 = -1)
                         Takes the non-zero argument and creates that many
                         columns or rows, respectively. Note: # Ues must be 
                         divisible by the non-zero index.
    - ...                     
    
        
    Returns
    -------
    Axis.

    """
    
    # Set useful variables
    n_ues = len(ue_list)
    n_y_vals_left = len(y_vals_left)
    n_y_vals_right = len(y_vals_right)
    
    if n_y_vals_left == 0:
        # We do not allow only right axis plot (for now)
        raise Exception('The left y axis must have values.')
    
    if n_y_vals_right == 0:
        n_y_vals = n_y_vals_left
        double_plot = False
    else:
        n_y_vals = n_y_vals_left + n_y_vals_right
        double_plot = True
    
    if len(ylim_left) == 1 and n_ues > 1:
        ylim_left *= n_ues
    
    if double_plot and len(ylim_right) == 1 and n_ues > 1:
        ylim_right *= n_ues
    
    # TODO: Set all of these other parameters properly to have fully 
    #       automated double plots
        
    if not no_ticks_left:
        no_ticks_left = [4 for i in ue_list]
    
    
    if double_plot and not no_ticks_right:
        no_ticks_right = [4 for i in ue_list]
        
    if y_axis_fonts:
        if double_plot and len(y_axis_fonts) != 2:
            raise Exception('One font per axis. Right and Left are 2 axis.')
        if not double_plot and len(y_axis_fonts) != 1:
            raise Exception('One font per axis. Right is 1. Do [font_scalar].')
        
    # Adjust and compute other variables from the inputs.
    if y_vals_left[0] is None:
        raise Exception('y_vals_left only has Nones in each UE...')
    
    if n_y_vals_right and y_vals_right[0] is None:
        raise Exception('y_vals only has Nones in each UE...')

    if linewidths == '':
        if same_axs:
            linewidths = [1 for ue in ue_list]
        else:
            linewidths = [1 for y_val in range(n_y_vals)]
    
    # Number of traces in the same plot
    n_opacity = n_y_vals if not same_axs else n_ues*n_y_vals
        
    if opacity == []:
        if tune_opacity:
            opacity = [1.2 / n_opacity] * n_opacity
        else:
            opacity = [1] * n_opacity
            
    if len(opacity) != n_opacity:
        raise Exception('Opacity length is not the correct one.')
    
    if y_labels_left == '':
        if same_axs:
            y_labels_left = [f'UE {ue}' for ue in ue_list]
        else:
            y_labels_left = ['' for y_val in range(n_y_vals_left)]
            if use_legend and not fill and not fill_label:
                raise Exception('No labels values provided in y_data_labels '
                      'or fill_labels. What should be in the legend?!')
    if double_plot:
        if y_labels_left == '':
            if same_axs:
                y_labels_right = [f'UE {ue}' for ue in ue_list]
            else:
                y_labels_right = ['' for y_val in range(n_y_vals_right)]
                if use_legend:
                    print('No labels values provided in y_data_labels. '
                          'What should be in the legend?!')
        
    if savefig:
        # Check file name
        if filename == '':
            if title != '':
                filename = title + ut.get_time()
            else:
                filename = ut.get_time()
        
        # Check file format
        if saveformat not in ['png', 'svg', 'pdf']:
            raise Exception('Unsupported save format.')
    
    
    # Create figure and axis
    r = width/height
    if same_axs:
        fig, axs = plt.subplots(tight_layout=True, 
                                figsize=(r*height*size, size/r*width))
    else:
        # Define number of subplots in each row and column
        
        # From input parameters:
        # -> set number of rows and number of columns
        if n1 != -1 and n2 != -1: 
            if n_ues != n1 + n2:
                raise Exception("Something doesn't add up... N1 and N2!")
        
        # -> set n1 rows and compute n2 columns
        if n1 != -1 and n2 == -1:
            if n_ues % n1 != 0:
                raise Exception("n_ues / n1 is not an integer.")
            else:
                n2 = int(n_ues / n1)
            
        # -> set n2 columns and n1 rows
        if n1 == -1 and n2 != -1:
            if n_ues % n2 != 0:
                raise Exception("n_ues / n2 is not an integer.")
            else:
                n1 = int(n_ues / n2)
                
        # -> derive n1 and n2 from the number of UEs
        else:
            if n_ues > 1:
                div_list = ut.divisors(n_ues)
                n1 = div_list[1]
                n2 = div_list[-2]
            else:
                n1 = 1
                n2 = 1
        
        # Create the subplots with a certain figure size, n1 rows and n2 cols
        fig, axs = plt.subplots(n1,n2, tight_layout=True, 
                                figsize=(r*height*size, size/r*width))
    
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    
    
    # Start the actual Plots
    for ue_idx in range(len(ue_list)):
        
        # Get ax index
        if same_axs:
            idx = 0
        else:
            if n2 > 1:
                aux = int(n_ues / 2)
                if ue_idx < aux:
                    idx = (0, ue_idx)
                else:
                    idx = (1, ue_idx - aux)
            else:
                idx = ue_idx
                
        ax1_handle = axs[idx]
        
        ax1_handle.tick_params(axis = 'both', which = 'major', labelsize = axes_tick_font_size)
        
        # Select x_data
        if x_vals.ndim == 1:
            x_data = x_vals
        else:
            x_data = x_vals[:,ue_idx]
        
        for y_idx in range(n_y_vals_left):
            if same_axs:
                if n_ues == 1:
                    p_idx = y_idx
                else:
                    p_idx = ue_idx # plot idx
            else:
                p_idx = y_idx
                ax1_handle.set_title(f'UE {ue_list[ue_idx]}')
            
            # Select y_data
            if y_vals_left[y_idx].ndim == 1:
                y_data = y_vals_left[y_idx]
            else:
                y_data = y_vals_left[y_idx][:,ue_idx]
            
            
            # Check sizes of x and y: trimming problem?
            if len(x_data) != len(y_data):
                raise Exception('x and y values have different dimensions. '
                                'Problem with trimming?')
            
            # Try to plot the left axis
            try:
                if plot_type_left == 'line':
                    ax1_handle.plot(x_data, y_data, alpha=opacity[p_idx], 
                                   linewidth=linewidths[p_idx], 
                                   label=y_labels_left[p_idx])
                elif plot_type_left == 'scatter':
                    ax1_handle.scatter(x_data, y_data,
                                       linewidth=linewidths[p_idx], 
                                       label=y_labels_left[p_idx])  
                elif plot_type_left == 'bar':
                    ax1_handle.bar(x_data, y_data)
                else:
                    raise Exception(f'No plot type named "{plot_type_left}".')
            except Exception as e:
                if type(e) == ValueError:
                    print('ERROR DESCRIPTION:')
                    print(e)
                    print('\nNote: a Value Error here usually happens '
                          'when you forget to put y inside a list: '
                          'plot(ues, x, [y])')
                else:
                    raise e
            
            ax1_handle.locator_params('y', nbins=no_ticks_left[ue_idx])
            
        for y_idx_right in range(n_y_vals_right):
        # if it is double, set a couple of things if they are not set in advance:
        # alpha=0.6
        # color_ax1 ='g'
        # color_ax2 ='b'
        # linewidth_ax2=1

            # Try to plot the right axis
            ax2_handle = ax1_handle.twinx()
            
            if same_axs:
                if n_ues == 1:
                    p_idx = y_idx
                else:
                    p_idx = ue_idx # plot idx
            else:
                p_idx = y_idx
                ax2_handle.set_title(f'UE {ue_list[ue_idx]}')
            
            # Select y_data
            if y_vals_right[y_idx].ndim == 1:
                y_data_right = y_vals_right[y_idx]
            else:
                y_data_right = y_vals_right[y_idx][:,ue_idx]
            
            try:
                if plot_type_right == 'line':                                            
                    ax2_handle.plot(x_data, y_data_right, alpha=opacity[p_idx], 
                                    linewidth=linewidths[p_idx], 
                                    label='' if not y_labels_right else
                                          y_labels_right[p_idx])
                elif plot_type_right == 'scatter':
                    ax2_handle.scatter(x_data, y_data_right)  
                elif plot_type_right == 'bar':
                    ax2_handle.bar(x_data, y_data_right)
                else:
                    raise Exception(f'No plot type named "{plot_type_right}".')
            except Exception as e:
                if type(e) == ValueError:
                    print('ERROR DESCRIPTION:')
                    print(e)
                    print('\nNote: a Value Error here usually happens '
                          'when you forget to put y inside a list: '
                          'plot(ues, x, [y])')
                else:
                    raise e
                    
            ax2_handle.locator_params('y', nbins=no_ticks_right[ue_idx])
                
        # Set X and Y labels 
        ax1_handle.set_xlabel(x_axis_label)
        ax1_handle.set_ylabel(y_axis_label[0])
        if double_plot:
            ax2_handle.set_ylabel(y_axis_label[1])
            
        # Set X and Y limits
        if xlim != []:
            if isinstance(xlim, tuple):
                ax1_handle.set_xlim(xlim)
            elif isinstance(xlim, list) and len(xlim) == n_ues:
                ax1_handle.set_xlim(xlim[ue_idx])
            else:
                raise Exception('xlim badly formatted: list of tuples, one '
                                'tuple for each ue')
        else:
            # ax1_handle.set_xlim([min(x_data)-1, max(x_data)+1])
            ax1_handle.autoscale(enable=True, axis='x', tight=tight_x_axis)
        
        if ylim_left != []:
            if isinstance(ylim_left, tuple):
                ax1_handle.set_ylim(ylim_left)
            elif isinstance(ylim_left, list) and len(ylim_left) == n_ues:
                ax1_handle.set_ylim(ylim_left[ue_idx])
            else:
                raise Exception('ylim badly formatted: list of tuples, one '
                                'tuple for each ue')
        else:
            ax1_handle.autoscale(enable=True, axis='y', tight=tight_y_axis)
        
        if double_plot and ylim_right != []:
            if ylim_right != []:
                if isinstance(ylim_left, tuple):
                    ax2_handle.set_ylim(ylim_right)
                elif isinstance(ylim_right, list) and len(ylim_right) == n_ues:
                    ax2_handle.set_ylim(ylim_right[ue_idx])
                else:
                    raise Exception('ylim badly formatted: list of tuples, one '
                                    'tuple for each ue')
            else:
                ax2_handle.autoscale(enable=True, axis='y', tight=tight_y_axis)
        
        
        # Set legend
        if use_legend and legend_inside:
            legend_handle = ax1_handle.legend(loc=legend_loc)
        
        # Set fonts of axis if specified:
        if y_axis_fonts:
            lab = ax1_handle.xaxis.get_label()
            lab.set_fontsize(y_axis_fonts[0])
            lab = ax1_handle.yaxis.get_label()
            lab.set_fontsize(y_axis_fonts[0])                        
            if double_plot:
                lab = ax1_handle.xaxis.get_label()
                lab.set_fontsize(y_axis_fonts[0])
                lab = ax2_handle.yaxis.get_label()
                lab.set_fontsize(y_axis_fonts[1])
        
        
        if fill:
            if isinstance(fill_var, str):
                raise Exception('no var provided for filling')
            
            low_lim, high_lim = ax1_handle.get_ylim()
            
            ax1_handle.fill_between(x_vals, 0, high_lim, 
                                    where=fill_var[:,ue_idx] > 0, 
                                    color=fill_color, 
                                    alpha=linewidths[-1],
                                    label=fill_label)
        
    if use_legend:
        if not legend_inside:
            handles, labels = ax1_handle.get_legend_handles_labels()
            legend_handle = fig.legend(handles, labels, loc=legend_loc, 
                                       fancybox=True, shadow=True, ncol=ncols)
        
        for legobj in legend_handle.legendHandles:
            legobj.set_linewidth(2.0)
    
    # Set Title
    if title != '':
        fig.suptitle(title)
    
    # Save Figure
    if savefig:
        if use_legend and not same_axs:
            fig.savefig(filename, format=saveformat,
                        bbox_extra_artists=(legend_handle,), 
                        bbox_inches='tight')
        else:
            plt.savefig(filename, format=saveformat, bbox_inches='tight')
            
        print(f'Saved: {filename}')
    
    # Display Figure
    plt.show()
    
    # Return Axis (for plot editing purposes)
    return axs


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
    
    ttis_changed = not np.array_equal(ttis, ttis_temp)
    # in the future, replace this loop for the proper python function 
    # compare the eleements of two lists.
    files_changed = not ut.elementwise_comparison_loop(set_of_files, 
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
                 1: [4], 1.1: [4], 1.2: [4],
                 2: [2,3], 2.1: [2,4,5,6], 2.15: [2,4,5,6], 
                 2.2: [2,3,4,8], 
                 2.3: [2,3], 2.4: [2,3,5,6],
                 3: [11], 3.1: [11], 3.2: [10, 15], 3.3: [10,12], 3.35: [10,12], 
                 3.4: [10,12], 3.45: [10,12], 3.5: [10,12], 3.55: [10,12], 
                 3.6: [12,20], 3.65: [12,20],
                 4.1: [9], 4.2: [9], 4.3: [4,9],
                 5.1: [7], 5.15: [7], 5.2: [7], 5.3: [7], 
                 5.4: [2,7], 5.5: [5,6,7], 5.6: [7], 5.65: [7],
                 7.1: [5,6], 7.2: [5,6], 7.3: [5,6],
                 7.35: [5,6], 7.4: [4,5,6], 7.5: [2,5,6],
                 9.1: [2,5,6,8], 9.2: [2,5,6,8], 9.3: [8,9], 9.4: [5,6,8],
                 10.1: [1], 10.15: [1], 10.2: [1], 10.25: [1], 
                 10.3: [1], 10.31: [1], 10.4: [1], 10.45: [1], 
                 10.5: [1], 10.55: [1], 10.6: [1], 10.65: [1, 4], 
                 10.7: [1], 10.8: [1], 10.9: [1], 10.11: [1],
                 11: [15], 11.1: [15], 11.2: [15], 11.3: [4], 11.4: [10,15],
                 13: [14],
                 14.1: [1], 14.2: [1], 
                 15: [18], 
                 16: [7], 16.1: [7], 16.2: [], 16.25: [], 16.3: [],
                 17: [], 17.01: [7,15], 17.02: [7,15], 17.03: [7,15], 
                 17.11: [7,15], 17.12: [7,15], 17.13: [7,15],
                 18.1: [2, 3, 15], 19.1: [2]
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
                    3: [11], 3.1: [6,11], 3.2: [], 3.3: [], 3.35: [], 
                    3.4: [5,7], 3.45: [5,7], 3.5: [5,7], 3.55: [5,7], 
                    3.6: [], 3.65: [7,8],
                    4.1: [], 4.2: [], 4.3: [],
                    5.1: [9], 5.15: [26], 5.2: [], 
                    5.3: [10], 5.4: [10], 5.5: [3,10], 5.6: [31], 5.65: [31],
                    7.1: [3], 7.2: [3,4], 7.3: [3,4],
                    7.35: [3,4], 7.4: [3], 7.5: [3],
                    9.1: [3], 9.2: [3], 9.3: [], 9.4: [3],
                    10.1: [12,14], 10.15: [12,14], 10.2: [12,15], 
                    10.25: [12,15], 10.3: [12,14,15], 10.31: [12,14,15], 
                    10.4: [12,13,14,15], 10.45: [12,13,14,15], 
                    10.5: [16], 10.55: [16], 10.6: [17], 10.65: [12,17], 
                    10.7: [14,15,16,17,18,19,20,21,22,23], 
                    10.8: [22,27], 10.9: [23,27], 
                    10.11: [22,23,27],
                    11: [24], 11.1: [], 11.2: [], 11.3: [25], 11.4: [],
                    13: [],
                    14.1: [], 14.2: [],
                    15: [28], 
                    16: [29,30], 16.1: [29,30],  16.2: [32], 16.25: [32], 16.3: [32],
                    17: [33,34], 17.01: [33,34,35,36], 17.02: [33,34,35,36], 
                    17.03: [33,34,35,36], 17.11: [35,36], 17.12: [35,36],
                    17.13: [35,36],
                    18.1: [37],
                    19.1: [37]
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
                print(f"Var '{v}' not saved in simulation.")
                continue
            
            sim_data_trimmed[f][v_idx] = \
                np.array(sim_data_loaded[f][v_idx])[trim_ttis[0]:trim_ttis[1]]
                
            # np.array(sim_data_loaded[f][v_idx])[trim_ttis[0]:trim_ttis[1]]
            # # Select the downlink ttis only
            #sim_data[f][v_idx] = np.delete(sim_data[v_idx, f], ul_ttis, axis=0)
    
    # UL tti formula, if first and last tti are multiples of 5.
    # ul_ttis = np.arange(0, last_tti - first_tti, 5) - 1
    # ul_ttis = ul_ttis[1:]
    
    pass

def compute_sim_data(plot_idx, l, ues, ttis, 
                     all_loadable_var_names, all_computable_var_names, 
                     vars_to_compute, vars_to_trim,
                     sim_data_trimmed, sim_data_computed, file_set,
                     vars_with_layers):
    
    # Setup some useful variables:
    n_ues = len(ues)
    n_ttis = len(ttis)
    n_files = len(file_set)
    
    
    # Let's handle first the single_trace computations
    # In this first computation phase, we compute variables per trace only.
    for f in range(n_files):
        # Current file simulation parameters
        f_sp = sim_data_trimmed[f][0]
        
        n_layers = f_sp.n_layers
        
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
                          f"{v_trim} wasn't trimmed or loaded successfully.")
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
            
            # COMPUTE INDEX 1: Running average bitrate 
            if var_to_compute == 'running_avg_bitrate' and \
               sim_data_computed[f][v] is None:
                # IDX 4 is for realised bit rate
                sim_data_computed[f][v] = \
                    np.cumsum(sim_data_trimmed[f][4], axis=0)
                for ue in range(n_ues):
                    sim_data_computed[f][v][:,ue,:] = \
                        (sim_data_computed[f][v][:,ue,:] / 
                         np.arange(n_ttis).reshape(n_ttis,1))
                
            # COMPUTE INDEX 2: Rolling avg bitrate
            if var_to_compute == 'rolling_avg_bitrate' and \
               sim_data_computed[f][v] is None:
                # IDX 4 is for realised bit rate
                sim_data_computed[f][v] = np.zeros((n_ttis, n_ues, n_layers))
                
                # TTIs of 1 GoP, interval over which to average the bit rate
                rolling_int = int(GoP / f_sp.FPS / f_sp.TTI_dur_in_secs)

                for ue in range(n_ues):
                    for l_i in range(n_layers):
                        sim_data_computed[f][v][rolling_int-1:,ue,l_i] = \
                            ut.moving_average(sim_data_trimmed[f][4][:,ue,l_i], 
                                              rolling_int)
            
            # COMPUTE INDEX 3: Instantaneous BLER 
            if var_to_compute in ['instantaneous_bler',
                                  'running_avg_bler'] and \
               sim_data_computed[f][v] is None:
                # IDX 5 is blocks_with_errors
                # IDX 6 is n_transport_blocks
                sim_data_computed[f][v] = (sim_data_trimmed[f][5] / 
                                           sim_data_trimmed[f][6] * 100)
                sim_data_computed[f][v] = np.nan_to_num(sim_data_computed[f][v])
                
            # COMPUTE INDEX 4: Running average BLER
            if var_to_compute == 'running_avg_bler' and \
               sim_data_computed[f][v] is None:
                # Requires instantaneous_bler, IDX 3 of computed
                sim_data_computed[f][v] = np.cumsum(sim_data_computed[f][3], 
                                                    axis=0)
                sim_data_computed[f][v] = (sim_data_computed[f][v] / 
                                           np.arange(n_ttis).reshape(n_ttis,1))
                
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
                                           np.log10(sim_data_trimmed[f][20]))
            
            # COMPUTE INDEX 9: Beam Formula Simple
            if var_to_compute == 'beam_formula_simple' and \
               sim_data_computed[f][v] is None:
                # IDX 7 is the beams_used
                sim_data_computed[f][v] = (sim_data_trimmed[f][7][:,:,:,0] + 
                                           sim_data_trimmed[f][7][:,:,:,1] * 10)
            
            
            # COMPUTE INDEX 10: Beam Sum
            if var_to_compute == 'beam_sum' and \
               sim_data_computed[f][v] is None:
                # IDX 7 is the beams_used
                sim_data_computed[f][v] = (sim_data_trimmed[f][7][:,:,:,0] + 
                                           sim_data_trimmed[f][7][:,:,:,1])
            
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
                sim_data_computed[f][13] = np.zeros([int(n_frames), n_ues],
                                                    dtype=int)
                sim_data_computed[f][13][0,:] = 1
                for frame_idx in range(int(n_frames)):
                    if frame_idx % GoP == 0:
                        sim_data_computed[f][13][frame_idx,:] = 1
            
            # COMPUTE INDEX 14: Average Packet Latency 
            if var_to_compute in ['avg_packet_lat',
                                  'avg_pck_lat_per_frame',
                                  'avg_pck_lat_per_frame_in_gop',
                                  'avg_pck_drop_rate_per_frame_in_gop'] and \
               sim_data_computed[f][14] is None:
                   
                sim_data_computed[f][14] = np.zeros([int(n_frames), n_ues])
                
                for ue in range(n_ues):
                    for per in range(int(n_periods)):
                        # Loop over frame indices (fi)
                        for fi in range(GoP):
                            # IDX 1 are the buffers
                            f_info = \
                                sim_data_trimmed[f][1][ue].frame_infos[per][fi]
                            sim_data_computed[f][14][fi][ue] = \
                                (f_info.avg_lat.microseconds / 1e3)
                                
            # COMPUTE INDEX 15: Average Packet Drop Rate
            if var_to_compute in ['avg_packet_drop_rate',
                                  'avg_pck_drop_rate_per_frame',
                                  'avg_pck_lat_per_frame_in_gop',
                                  'avg_pck_drop_rate_per_frame_in_gop'] and \
               sim_data_computed[f][15] is None:
                   
                sim_data_computed[f][15] = np.zeros([int(n_frames), n_ues])
                
                # Compute packet success percentages, average latencies and 
                # drop rates
                for ue in range(n_ues):
                    for per in range(int(n_periods)):
                        for frm in range(GoP):
                            
                            f_info = \
                                sim_data_trimmed[f][1][ue].frame_infos[per][frm]
                            packets_sent = f_info.successful_packets
                            dropped_packets = f_info.dropped_packets
                            total_packets = \
                                packets_sent + dropped_packets
                            
                            # print('UE:', ue, 'Period:', per, \
                            #       'GoP-Frame', frm,\
                            #       'Total packets:',total_packets)
                                
                            frame_idx = per * GoP + frm
                            
                            try:
                                sim_data_computed[f][15][frame_idx][ue] = \
                                    dropped_packets / total_packets * 100
                            except ZeroDivisionError:
                                print('Error!!!!')
                                print('Solve this error if it occurs again.')
                    
                
                
                
            # COMPUTE INDEX 16: Average Packet Latency across all frames
            if var_to_compute == 'avg_pck_lat_per_frame' and \
               sim_data_computed[f][v] is None:
                
                # IDX 14 is the avg_packet_latency
                aux = np.zeros([n_frames, n_ues])
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
                  packets_sent_full = np.zeros(n_ues)
                  dropped_packets_full = np.zeros(n_ues)
                  total_packets_full = np.zeros(n_ues)
                  pdr = np.zeros(n_ues)
                  for ue in range(n_ues):
                    for per in range(int(n_periods)):
                        for frm in range(GoP):
                            
                            f_info = \
                                sim_data_trimmed[f][1][ue].frame_infos[per][frm]
                            packets_sent_full[ue] += f_info.successful_packets
                            dropped_packets_full[ue] += f_info.dropped_packets
                           
                            
                            # print('UE:', ue, 'Period:', per, \
                            #       'GoP-Frame', frm,\
                            #       'Total packets:',total_packets)
                                
                            frame_idx = per * GoP + frm
                    total_packets_full[ue] = \
                                packets_sent_full[ue] + dropped_packets_full[ue] 
                    pdr[ue] = dropped_packets_full[ue] / total_packets_full[ue] * 100
                  
                  try:
                        sim_data_computed[f][v] = pdr
                                    
                  except ZeroDivisionError:
                                print('Error!!!!')
                                print('Solve this error if it occurs again.')    
                   
                # # IDX 15 is the avg_packet_drop_rate
                # aux = np.zeros([int(n_frames), n_ues])
                # aux[:] = sim_data_computed[f][15][:] 
                
                # # Scale up I frames: (IDX 13 is the I frame indices )
                # aux[sim_data_computed[f][13], :] *= 1 / f_sp.IP_ratio
                
                # # Don't try to understand this math... Basically we needed to 
                # # weigh the number of packets of each frame and divide by the 
                # # number of frames.
                # b = (1 / f_sp.IP_ratio) * ((GoP - 1) * f_sp.IP_ratio + 1)
                
                # sim_data_computed[f][v] = \
                #     np.round(np.sum(aux, 0) / 
                #              (len(sim_data_computed[f][12]) * b / GoP), 2)
                    
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
                sim_data_computed[f][9] = (sim_data_trimmed[f][7][:,:,:,0] + 
                                           sim_data_trimmed[f][7][:,:,:,1] * 10)
                
                # Copy the old beam_formula
                sim_data_computed[f][v] = sim_data_computed[f][9]
                
                # Process the formula to make it smoother
                
                # Keep the formula constant if the next value is 0
                for tti in range(n_ttis):
                    for ue in range(n_ues):
                        if sim_data_computed[f][v][tti, ue] == 0:
                            sim_data_computed[f][v][tti, ue] = \
                                sim_data_computed[f][v][tti-1, ue]
                    
                # address first zeros too
                for ue in range(n_ues):
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
        
            
            # COMPUTE INDEX 28: Power per beam in the GoB
            if var_to_compute == 'power_per_gob_beam' and \
               sim_data_computed[f][v] is None:
                n_beams = sim_data_trimmed[f][18].shape[-1]
                new_shape = (n_beams, n_ttis, n_ues)
                sim_data_computed[f][v] = np.zeros(new_shape)
                
                for i in range(n_beams):
                    sim_data_computed[f][v][i,:,:,:] = \
                        sim_data_trimmed[f][18][:,:,:,i]
                
            # COMPUTE INDEX 29 & 30: x and y projections of best beams
            if var_to_compute in ['x_projection_best_beam',
                                  'y_projection_best_beam'] and \
               (sim_data_computed[f][29] is None or 
                sim_data_computed[f][30] is None):
                # Project plane will be at ue_h
                bs_h = 3   # bs height
                ue_h = 1.6 # assumes the UEs are at the same height
                h = bs_h - ue_h # height between bs to ue
                
                # x uses elevation
                sim_data_computed[f][29] = \
                    h * np.tan(np.deg2rad(sim_data_trimmed[f][7][:,:,:,1]))
                # y uses azimuth (that's how the BS is oriented)
                sim_data_computed[f][30] = \
                    h * np.tan(np.deg2rad(sim_data_trimmed[f][7][:,:,:,0]))
                
            # COMPUTE INDEX 31: Beam Switch
            if var_to_compute == 'beam_switch' and \
               sim_data_computed[f][v] is None:
                # often beam_powers is not available... use another method.
                # max_beam_idx = np.argmax(beam_powers[:,0,:], 1)
                # IDX 7 is beams_used
                max_beam_idx = (sim_data_trimmed[f][7][:,:,:,0] + 
                                sim_data_trimmed[f][7][:,:,:,1] * 10000)
                # like beam formula simple, but more robust
                
                sim_data_computed[f][v] = np.zeros((n_ttis, n_ues, n_layers))
                
                # Put to one when there is a difference between beams used
                sim_data_computed[f][v][1:,:,:] = np.diff(max_beam_idx, axis=0)
                sim_data_computed[f][v][sim_data_computed[f][v] != 0] = 1
            
            # COMPUTE INDEX 32: x and y projections of all beams in the GoB
            if var_to_compute in ['xy_projection_all_gob'] and \
               sim_data_computed[f][v] is None:
                
                # Project plane will be at ue_h
                bs_h = 3   # bs height
                ue_h = 1.6 # assumes the UEs are at the same height
                h = bs_h - ue_h # height between bs to ue
                
                sim_data_computed[f][v] = np.zeros((f_sp.gob_n_beams,2))
                
                # x uses elevation
                sim_data_computed[f][v][:,0] = \
                    h * np.tan(np.deg2rad(f_sp.gob_directions[1,:]))
                # y uses azimuth (that's how the BS is oriented)
                sim_data_computed[f][v][:,1] = \
                    h * np.tan(np.deg2rad(f_sp.gob_directions[0,:]))

            # COMPUTE INDEX 33 & 34: User position and orientation ready to plot
            if var_to_compute in ['user_pos_for_plot',
                                  'user_ori_for_plot'] and \
               (sim_data_computed[f][33] is None or
                sim_data_computed[f][34] is None):
                
                # sim_data_computed[f][33] is [n_ue, 3, n_ttis] 
                # sim_data_computed[f][34] is [n_ue, 3, n_ttis, 2]

                # Track processing - the track was compressed
                n_ttis_compressed = f_sp.pos_backup.shape[-1]
                
                # Position
                initial_pos = np.reshape(f_sp.initial_pos_backup,
                                                  (n_ues, 3, 1))
                sim_data_computed[f][33] = (f_sp.pos_backup + initial_pos)

                # Orientation
                # Length of orientation vector [m]
                l_ori = 0.8

                theta = np.zeros((n_ues, 3, n_ttis_compressed))
                phi = np.zeros((n_ues, 3, n_ttis_compressed))
                target_pos = np.zeros((n_ues, 3, n_ttis_compressed))

                # Create point that is l_ori away from the current position,
                # in the direction of the orientation. We just have to plot
                # a vector between the current position and the target position
                for ue in range(n_ues):
                    theta = f_sp.ori_backup[ue, 1, :] * (-1) + np.pi / 2
                    phi = f_sp.ori_backup[ue, 2, :]
                    x_target_pos = (sim_data_computed[f][33][ue, 0, :] + 
                                    l_ori * np.sin(theta) * np.cos(phi)) 
                    y_target_pos = (sim_data_computed[f][33][ue, 1, :] + 
                                    l_ori * np.sin(theta) * np.sin(phi))
                    z_target_pos = (sim_data_computed[f][33][ue, 2, :] + 
                                    l_ori * np.cos(theta))
                    target_pos[ue, :, :] = \
                        np.vstack((x_target_pos, y_target_pos, z_target_pos))

                # Create the vector, ready to be plotted
                ori_line = np.zeros((n_ues, 3, n_ttis_compressed, 2))
                for ue in range(n_ues):
                    ori_line[ue, 0, :, :] = \
                        np.hstack((np.reshape(sim_data_computed[f][33][ue,0,:], 
                                             (-1, 1)), 
                                  np.reshape(target_pos[ue,0,:], (-1, 1))))
                    ori_line[ue, 1, :, :] = \
                        np.hstack((np.reshape(sim_data_computed[f][33][ue,1,:], 
                                             (-1, 1)), 
                                   np.reshape(target_pos[ue,1,:], (-1, 1))))
                    ori_line[ue, 2, :, :] = \
                        np.hstack((np.reshape(sim_data_computed[f][33][ue,2,:], 
                                             (-1, 1)), 
                                   np.reshape(target_pos[ue,2,:], (-1, 1))))
                
                sim_data_computed[f][34] = ori_line

            # COMPUTE INDEX 35: Beam details (correct HPBWs, ...)
            if var_to_compute == 'individual_beam_gob_details' and \
               sim_data_computed[f][v] is None:
                pass
                # folder = sim_data_trimmed[0][0].precoders_folder + '\\'
                folder = r'C:\Users\Srijan\Documents\SXRSIMv3\Matlab\precoders' + '\\'
                # file = 'beam_details_4_4_-60_60_12_0_-60_60_12_0_pol_1.mat'
                file = 'beam_details_4_4_4_4_pol_3_RI_1_ph_1.mat'
                print(f'Loading beam details file: {file}')
                
                # [121][6]:
                # 121 beams x (HPBW-AZ, HPBW-EL, 
                #              max direction (AZ), max direction (EL),
                #              Linear Amplitude gain, Logarithmic power gain)
                try:
                    sim_data_computed[f][v] = \
                        scipy.io.loadmat(folder + file)['beam_details']
                except FileNotFoundError:
                    raise Exception('File not found. Did you remember the '
                                    'beam details need to be generated '
                                    'separately?')

            # COMPUTE INDEX 36: When the UEs are not scheduled, keep the same
            #                   beam, for visualization purposes. 
            if var_to_compute == 'beams_processed' and \
               sim_data_computed[f][v] is None:
                # if not scheduled, keep the same beam (it will only change
                # color in the plots)
                # IDX 7 is beams_used
                sim_data_computed[f][v] = sim_data_trimmed[f][7][:,:,l,:]
                
                # IDX 15 is scheduled_ues
                for tti in range(1, n_ttis):
                    for ue in range(n_ues):
                        if sim_data_trimmed[f][15][tti][ue] == 0:
                            sim_data_computed[f][v][tti,ue,:] = \
                                sim_data_computed[f][v][tti-1,ue,:]

            # COMPUTE INDEX 37: Avg. SINR across time, per UE
            if var_to_compute == 'avg_sinr' and \
               sim_data_computed[f][v] is None:
                sim_data_computed[f][v] = np.zeros([n_ues, n_layers])
                
                for ue in range(n_ues):
                    for tti in range(0, n_ttis-1):
                        if sim_data_trimmed[f][15][tti][ue] == 0:
                            sim_data_trimmed[f][3][tti, ue, :] = np.nan
                
                for ue in range(n_ues):
                    for l_i in range(n_layers):
                        sim_data_computed[f][v][ue][l_i] = \
                        np.nanmean(sim_data_trimmed[f][3][:, ue, l_i])
                # print(sim_data_trimmed[f][3][:, 0, 0])  
                # print(sim_data_trimmed[f][3][:, 1, 0])
                # print(sim_data_trimmed[f][3][:, 2, 0])
                # print(sim_data_trimmed[f][3][:, 3, 0])
            
            # COMPUTE INDEX XX: 
            if var_to_compute == 'xxxxxxx' and \
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


def plot_sim_data(plot_idx, file_set, l, ues, ttis, x_vals, 
                  sim_data_trimmed, sim_data_computed, results_filename, 
                  base_folder, save_fig, save_format='svg'):
    
    """
        THE MEANING OF EACH PLOT INDEX IS IN PLOTS_PHASE1.PY.
    """
    
    x_label_time = 'Time [s]'
    n_ues = len(ues)
    n_files = len(file_set)
    n_ttis = len(ttis)
    
    ########################### PART OF PLOTING ##############################
    for f in range(n_files):
        f_sp = sim_data_trimmed[f][0]
        
        curr_file = file_set[f]
        
        # File naming
        freq_save_suffix = '_' + str(round(f_sp.freq / 1e9,1)) + 'ghz'
        stats_folder = curr_file.split('\\')[-2]
        save_folder = base_folder + stats_folder + '\\'
        raw_file_name = 'IDX-' + str(plot_idx) + freq_save_suffix
        file_name = save_folder + raw_file_name
        file_name += '.' + save_format
        
        if save_fig and not ut.isdir(save_folder):
            ut.makedirs(save_folder)
                    
        # Avg. channel across time (sum over the PRBs, and antenna elements)
        if plot_idx == 0.1:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][16]],
                         x_axis_label=x_label_time, 
                         y_axis_label=['Power [dBW]'],
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        # The two indicies below require the channel_per_prb variable
        # Channel across time for many PRBs
        if plot_idx == 0.2:
            pass
        
        # Channel across PRBs for one time (tti)
        if plot_idx == 0.3:
            pass
        
        
        # Instantaneous Realised Bitrate
        if plot_idx == 1:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][4][:,:,l]], 
                         x_axis_label=x_label_time, 
                         y_axis_label=['Realised bit rate [Mbps]'], 
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        
        # Inst. vs Running avg bitrate 
        if plot_idx == 1.1:
            plot_for_ues(ues, x_vals, 
                         [sim_data_trimmed[f][4][:,:,l], sim_data_computed[f][1]], 
                         x_axis_label=x_label_time, 
                         y_axis_label=['Realised bit rate [Mbps]'], 
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
    
    
        # Inst. vs Rolling avg bitrate 
        if plot_idx == 1.2:
            # Make constant line
            marker_line = np.ones(sim_data_trimmed[f][4][:,:,l].shape) * 80
        
            if n_ues == 1 and f_sp.avg_bitrate_dl < 400:
                # (Single-user case - normal bitrate)
                plot_for_ues(ues, x_vals,
                             [sim_data_trimmed[f][2][:,:,l], sim_data_computed[f][2]], 
                             x_axis_label=x_label_time, 
                             y_axis_label=['Bit rate [Mbps]'], 
                             title='', linewidths=[0.8, 2], 
                             y_labels_left=['Instantaneous', 
                                            'Rolling avg.\nover GoP duration'], 
                             use_legend=True, legend_inside=True, 
                             legend_loc=(0.64,0.2), ncols=1, size=1, 
                             filename=file_name, savefig=save_fig, 
                             same_axs=True) 
            
            if n_ues == 1 and f_sp.avg_bitrate_dl > 400:
                # (Single-user case - full buffer)
                plot_for_ues(ues, x_vals, 
                             [sim_data_trimmed[f][4][:,:,l], sim_data_computed[f][2]], 
                             x_axis_label=x_label_time, 
                             y_axis_label=['Bit rate [Mbps]'], 
                             title='', linewidths=[0.8, 2], 
                             y_labels_left=['Instantaneous', 
                                            'Rolling avg.\nover GoP duration'], 
                             use_legend=True, legend_inside=True, 
                             legend_loc=(0.645,.2), 
                             size=1, width=6.4, height=4, 
                             filename=file_name,
                             savefig=save_fig, same_axs=True) 
            
            if n_ues > 1:
                # (Multi-user case)
                plot_for_ues(ues, x_vals, [sim_data_trimmed[f][4][:,:,l], 
                                           sim_data_computed[f][2], 
                                           marker_line], 
                             x_axis_label=x_label_time, 
                             y_axis_label=['Bit rate [Mbps]'], 
                             title='', linewidths=[0.3, 1.5, 1], 
                             y_labels_left=['Instantaneous', 
                                            'Rolling avg. over GoP duration', 
                                            'Application Bit rate'], 
                             ylim_left=(-8, 240),
                             use_legend=True, ncols=3,
                             size=1.3, filename=file_name, 
                             savefig=save_fig) 
            
                
        # SINR (Multi-user case)
        if plot_idx == 2:
            plot_for_ues(ues, x_vals, 
                         [sim_data_trimmed[f][2][:,:,l], sim_data_trimmed[f][2][:,:,l]], 
                         x_axis_label=x_label_time, y_axis_label=['SINR [dB]'], 
                         y_labels_left=['Estimated', 'Experienced'], 
                         use_legend=True, ncols=2, size=1.3,filename=file_name, 
                         savefig=save_fig)
    
        # SINR vs BLER: only when there are active transmissions (single_user)
        if plot_idx == 2.1:
            plot_for_ues([1], x_vals, 
                         [sim_data_trimmed[f][2][:,:,l]],
                         [sim_data_computed[f][3]], 
                         x_axis_label=x_label_time, 
                         y_axis_label=['Experienced SINR [dB]', 
                                       'BLER [%]'],
                         linewidths=[1,0.4,0.15], 
                         ylim_left=[(15,22)], no_ticks_left=[5],
                         y_axis_fonts=[17,17], fill=True, 
                         fill_var=sim_data_trimmed[f][4][:,:,l], 
                         use_legend=True,
                         legend_loc=(1.02,.955), 
                         legend_inside=False,
                         fill_label='Active\ntransmissions',
                         width=7.8, height=4.8, size=1.2,
                         filename=file_name, 
                         savefig=save_fig)
        
        # SINR vs BLER: with active transmissions (multi-user)
        if plot_idx == 2.15:
            plot_for_ues(ues, x_vals, 
                         [sim_data_trimmed[f][2][:,:,l]],
                         [sim_data_computed[f][3]], 
                         x_axis_label=x_label_time, 
                         y_axis_label=['Experienced SINR [dB]', 
                                       'BLER [%]'],
                         linewidths=[1,0.4,0.15], 
                         ylim_left=[(15,22)] * n_ues, 
                         no_ticks_left=[5] * n_ues,
                         fill=True, fill_var=sim_data_trimmed[f][4][:,:,l], 
                         fill_label='Active\ntransmissions',
                         filename=file_name, 
                         savefig=save_fig)
            
            
         # SINR vs OLLA: with active transmissions (multi-user)
        if plot_idx == 2.2:
            # TODO: fix legend when the only legend is the fill label.
            print('why does the legend of the fill label not appear?')
            plot_for_ues(ues, x_vals, 
                         [sim_data_trimmed[f][2][:,:,l]], 
                         [sim_data_trimmed[f][3][:,:,l]], 
                         x_axis_label=x_label_time, 
                         y_axis_label=['Experienced SINR [dB]', 
                                       '$\Delta_{OLLA}$'],
                         linewidths=[1,1,0.15], 
                         y_axis_fonts=[13,16], fill=True, 
                         fill_var=sim_data_trimmed[f][4][:,:,l], 
                         use_legend=True,
                         legend_loc=(1.02,.955), 
                         legend_inside=False,
                         fill_label='Active\ntransmissions',
                         width=7.8, height=4.8, size=1.2,
                         filename=file_name, 
                         savefig=save_fig)
    
        # SINR difference (realised - estimated)
        if plot_idx == 2.3:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][0]], 
                         x_axis_label=x_label_time, 
                         y_axis_label=['SINR diff [dB]'], 
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
            
        # SINR difference vs BLER
        if plot_idx == 2.4:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][0]], 
                                [sim_data_computed[f][3]],
                                x_axis_label=x_label_time, 
                                y_axis_label=['SINR diff [dB]', 'BLER [%]'],
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
            
            
        # Signal power variation across PRBs
        if plot_idx == 3:
            if not f_sp.save_per_prb_sig_pow:
                print('Per PRB signal power not saved during SIM!')
                continue
            
            # antenna index: 
            tti_idx = 3
            
            if len(sim_data_computed[f][11]) > 1:
                plt_type = 'line'
            else:
                plt_type = 'scatter'
                
            plot_for_ues(ues, sim_data_computed[f][11], 
                         [sim_data_trimmed[f][11][tti_idx,:,l,:].T], 
                         x_axis_label='Frequency [Hz]', 
                         y_axis_label=['Power [W]'], 
                         savefig=save_fig, plot_type_left=plt_type)
                
        
        # Signal power variation across PRBs in dB
        if plot_idx == 3.1:
            if not f_sp.save_per_prb_sig_pow:
                print('Per PRB signal power not saved during SIM!')
                continue
            
            # antenna index: 
            tti_idx = 3
            
            plot_for_ues(ues, sim_data_computed[f][11], 
                         [sim_data_computed[f][6]], 
                         x_axis_label='Frequency [Hz]', 
                         y_axis_label=['Power [dB]'], 
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
            
        # Signal power 
        if plot_idx == 3.2:
            # Plot signal power variation across time
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][10][:,:,l]], 
                         x_axis_label=x_label_time, 
                         y_axis_label=['Power [W]'], 
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
            
    
        # Signal power vs interference (Watt) [single axis]
        if plot_idx == 3.3:        
            plot_for_ues(ues, x_vals, 
                         [sim_data_trimmed[f][10][:,:,l], sim_data_trimmed[f][12][:,:,l]], 
                         x_axis_label=x_label_time, y_axis_label=['[W]'], 
                         y_labels_left=['Signal', 'Interference'],
                         use_legend=True, legend_loc='lower center', ncols=2,
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)

        # Signal power vs interference (Watt) [double axis]
        if plot_idx == 3.35:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][10][:,:,l]], 
                                [sim_data_trimmed[f][12][:,:,l]], 
                                x_axis_label=x_label_time, 
                                y_axis_label=['Signal Power [W]', 
                                              'Interference Power [w]'],
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format) 

        # Signal power vs interference (dBw) [single axis]
        if plot_idx == 3.4:
            plot_for_ues(ues, x_vals, 
                         [sim_data_computed[f][5], sim_data_computed[f][7]], 
                         x_axis_label=x_label_time, y_axis_label=['[dBw]'], 
                         y_labels_left=['Signal', 'Interference'],
                         use_legend=True, legend_loc='lower center', ncols=2,
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        
        # Signal power vs interference (dBw) [double axis]
        if plot_idx == 3.45:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][5]], 
                                [sim_data_computed[f][7]], x_label_time, 
                                ['Sig. Power [dBw]', 'Int. Power [dBw]'],
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
                
        # Signal power vs interference (dBm) [single]
        if plot_idx == 3.5:
            plot_for_ues(ues, x_vals,[sim_data_computed[f][5] + 30, 
                                      sim_data_computed[f][7] + 30], 
                         x_axis_label=x_label_time, 
                         y_axis_label=['Power [dBm]'],
                         y_labels_left=['Signal', 'Interference'],
                         use_legend=True, legend_loc='lower center', ncols=2,
                         size=1.3, width=6.4, height=4,
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
            
        # Signal power vs interference (dBm) [Double]
        if plot_idx == 3.55:
            plot_for_ues(ues, x_vals, 
                                [sim_data_computed[f][5] + 30], 
                                [sim_data_computed[f][7] + 30], 
                                x_axis_label=x_label_time, 
                                y_axis_label=['Signal Power [dBm]', 
                                              'Interference Power [dBm]'],
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
    
        
        # Estimated vs Realised interference
        if plot_idx == 3.6:
            plot_for_ues(ues, x_vals, 
                         [sim_data_trimmed[f][20][:,:,l], sim_data_trimmed[f][12][:,:,l]], 
                         x_axis_label=x_label_time, 
                         y_axis_label=['Interference Power [W]'],  
                         y_labels_left=['Estimated', 'Realised'],
                         use_legend=True, legend_loc='lower center', ncols=2,
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
         
        # Estimated vs Realised interference [dB]
        if plot_idx == 3.65:
            plot_for_ues(ues, x_vals, 
                         [sim_data_computed[f][8], sim_data_computed[f][7]], 
                         x_axis_label=x_label_time, 
                         y_axis_label=['Interference Power [dB]'],
                         y_labels_left=['Estimated', 'Realised'],
                         use_legend=True,
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
                            
        # MCS same axs
        if plot_idx == 4.1:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][9][:,:,l]], 
                         x_axis_label=x_label_time, 
                         y_axis_label=['MCS index'], ylim_left=(0.5, 15.5), 
                         use_legend=True, legend_inside=True, 
                         legend_loc="lower right",
                         ncols=1, size=1.3, same_axs=True,
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
            
        # MCS diff axs
        if plot_idx == 4.2:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][9][:,:,l]], 
                         x_axis_label=x_label_time, 
                         y_axis_label=['MCS index'],
                         linewidths=[.4,.4,.4,.4], 
                         y_labels_left=['UE 0','UE 1','UE 2','UE 3'], 
                         ylim_left=(6.5, 15.5),
                         ncols=1, size=1.3, 
                         use_legend=True, legend_inside=True, 
                         legend_loc="lower right",
                         same_axs=False,
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
    
        # MCS and instantaneous bitrate per UE
        if plot_idx == 4.3:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][9][:,:,l]], 
                                [sim_data_trimmed[f][4][:,:,l]], 
                                x_label_time, 
                                y_axis_label=['MCS index', 
                                              'Bit rate [Mbps]'],
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
        
        # Beams: best beam per user (filtered: prevents going back to zero when
        #        the UE is no scheduled. One plot per UE    
        if plot_idx == 5.1:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][9]], 
                         title='Formula: azi + ele x 10',
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        # Beam formula processed (filtered and smoother) !
        if plot_idx == 5.15:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][26]], 
                         title='Formula: azi + ele x 10',
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
            
        # Beams filtered: doublePlot per UE for azi and elevation values.
        if plot_idx == 5.2:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][7][:,:,l,0]], 
                                [sim_data_trimmed[f][7][:,:,l,1]],
                                x_axis_label=x_label_time, 
                                y_axis_label=['Azimuth [º]', 'Elevation[º]'],
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
    
        # Beam sum: used to notice beam switching easily
        if plot_idx == 5.3:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][10]],
                         x_axis_label=x_label_time, 
                         y_axis_label=['Azimuth + Elevation [º]'],
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        # Beam sum vs SINR
        if plot_idx == 5.4:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][10]], 
                                [sim_data_trimmed[f][2][:,:,l]], 
                                x_axis_label=x_label_time,
                                y_axis_label=['Azi. + El. [º]', 'SINR [dB]'],
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
        
        # Beam sum vs BLER\
        if plot_idx == 5.5:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][10]], 
                                [sim_data_computed[f][3]], 
                                x_axis_label=x_label_time,
                                y_axis_label=['Azi. + El. [º]', 'BLER [%]'],
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
        
        # Beam Switch: single plot
        if plot_idx == 5.6:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][31]],
                         use_legend=True, legend_inside=True, 
                         legend_loc="lower right", same_axs=True,
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        # Beam Switch: multiple plots
        if plot_idx == 5.65:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][31]],
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        # BLER: Instantaneous
        if plot_idx == 7.1:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][3]],
                         x_axis_label=x_label_time, 
                         y_axis_label=['Blocks with errors [%]'],
                         savefig=save_fig)
        
        # BLER: Running Average
        if plot_idx == 7.2:       
            plot_for_ues(ues, x_vals, [sim_data_computed[f][4]], 
                         x_axis_label=x_label_time, 
                         y_axis_label=['Running Avg. BLER [%]'],
                         width=6.4, height=4.8, size=1.3,
                         same_axs=False,
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        # BLER: Instantaneous + Running Average
        if plot_idx == 7.3:
            plot_for_ues(ues, x_vals,
                         [sim_data_computed[f][3], sim_data_computed[f][4]],
                         x_axis_label=x_label_time, 
                         y_axis_label=['Running average of BLER [%]'],
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)

        # BLER: Instantaneous + Running Average [double axis]
        if plot_idx == 7.35:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][3]], 
                                [sim_data_computed[f][4]],
                                x_axis_label=x_label_time, 
                                y_axis_label=['Inst. BLER [%]', 
                                              'Run. Avg. BLER [%]'],
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
            
                    
            
        # BLER: instantaneous BLER and realised bitrate
        if plot_idx == 7.4:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][4][:,:,l]],
                                [sim_data_computed[f][3]],
                                x_axis_label=x_label_time, 
                                y_axis_label=['Inst. bitrate [Mbps]', 
                                              'BLER [%]'],
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
            
            
        # BLER: Instantaneous vs realised SINR
        if plot_idx == 7.5:
            plot_for_ues(ues, x_vals, [sim_data_computed[f][3]], 
                                [sim_data_trimmed[f][2][:,:,l]], 
                                x_axis_label=x_label_time,
                                y_axis_label=['BLER [%]', 'SINR [dB]'],
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
            
        # OLLA (single-user) with Bitrate for grey areas
        if plot_idx == 9.1:
            # ONE UE ONLY:
            plot_for_ues([2], x_vals, [sim_data_computed[f][3]],
                                [sim_data_trimmed[f][8]], 
                                x_axis_label=x_label_time, 
                                y_axis_label=['BLER [%]', '$\Delta_{OLLA}$'],
                                linewidths=[0.2,1,0.15], 
                                y_axis_fonts=[13,16], fill=True, 
                                fill_var=sim_data_trimmed[f][2][:,:,l], 
                                use_legend=True,
                                legend_loc=(1.02,.955), 
                                legend_inside=False,
                                fill_label='Active\ntransmissions',
                                width=7.8, height=4.8, size=1.2,
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
            
        # OLLA (multi-user) with active transmission (bitrate > 0)
        if plot_idx == 9.2:
            # MULTIPLE UEs
            plot_for_ues(ues, x_vals, [sim_data_computed[f][3]],
                                [sim_data_trimmed[f][8]], 
                                x_axis_label=x_label_time, 
                                y_axis_label=['BLER [%]', '$\Delta_{OLLA}$'],
                                linewidths=[0.2,1,0.15], fill=True, 
                                fill_var=sim_data_trimmed[f][2][:,:,l],
                                fill_label='Active\ntransmissions',
                                width=7.8, height=4.8, size=1.2,
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
        
        # OLLA: MCS vs olla
        if plot_idx == 9.3:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][9][:,:,l]], 
                         [sim_data_trimmed[f][8]], 
                         x_axis_label=x_label_time, 
                         y_axis_label=['CQI IDX', '$\Delta_{OLLA}$'], 
                         linewidths=[0.2,0.9],
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
         
        # OLLA: inst. bler vs olla        
        if plot_idx == 9.4:
            plot_for_ues([0, 2], x_vals, [sim_data_computed[f][3]], 
                                [sim_data_trimmed[f][8]], 
                                x_axis_label=x_label_time, 
                                y_axis_label=['BLER [%]', '$\Delta_{OLLA}$'], 
                                linewidths=[0.2,0.9],
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
        
       
        
        # LATENCY and DROP RATE
        # avg latency across frames 
        if plot_idx == 10.1:
            plot_for_ues(ues, sim_data_computed[f][12], 
                         [sim_data_computed[f][14]],
                         x_axis_label='Frame index', 
                         y_axis_label=['Avg. latency [ms]'], 
                         linewidths=[0.7,0.6],
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        # avg latency across frames (bar chart)
        if plot_idx == 10.15:
            plot_for_ues(ues, sim_data_computed[f][12], 
                         [sim_data_computed[f][14]],
                         x_axis_label='Frame index', 
                         y_axis_label=['Avg. latency [ms]'],
                         linewidths=[0.7,0.6], 
                         plot_type_left='bar',
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        # avg drop rate across frames
        if plot_idx == 10.2:
            plot_for_ues(ues, sim_data_computed[f][12], 
                         [sim_data_computed[f][15]],  
                         x_axis_label='Frame index', 
                         y_axis_label=['Drop rate [%]'], 
                         linewidths=[0.7,0.6],
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        # avg drop rate across frames (bar chart)
        if plot_idx == 10.25:
            plot_for_ues(ues, sim_data_computed[f][12], 
                         [sim_data_computed[f][15]],  
                         x_axis_label='Frame index', 
                         y_axis_label=['Drop rate [%]'],
                         linewidths=[0.7,0.6], plot_type_left='bar',
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        # avg latency vs drop rate across frames (no I vs P distinction)
        if plot_idx == 10.3:    
            plot_for_ues(ues, sim_data_computed[f][12], 
                                [sim_data_computed[f][14]], 
                                [sim_data_computed[f][15]], 
                                x_axis_label='Frame index', 
                                y_axis_label=['Avg. latency [ms]', 
                                              'Drop rate [%]'], 
                                linewidths=[0.7,0.6],
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
        
        # Same as 10.3 but showing the ticks and limits.
        if plot_idx == 10.31:    
            plot_for_ues(ues, sim_data_computed[f][12], 
                                [sim_data_computed[f][14]], 
                                [sim_data_computed[f][15]], 
                                x_axis_label='Frame index', 
                                y_axis_label=['Avg. latency [ms]', 
                                              'Drop rate [%]'], 
                                linewidths=[0.7,0.6],
                                ylim_left=[[0,5],[0,0.6],[0,5],[0,0.6]],
                                ylim_right=[[-0.05,0.05],[-0.05,0.05],
                                            [-0.05,0.05],[-0.05,0.05]],
                                no_ticks_left=[4,4,4,4],
                                no_ticks_right=[4,4,4,4],
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
        
        # Average latency and drop rate with I frame markings: line
        if plot_idx == 10.4:
            plot_for_ues(ues, sim_data_computed[f][12], 
                                [sim_data_computed[f][14]],
                                [sim_data_computed[f][15]], 
                                x_axis_label='Frame index', 
                                y_axis_label=['Avg. latency [ms]',
                                              'Drop rate [%]'],
                                linewidths=[0.6,.6,0.4], 
                                y_axis_fonts=[14,14], fill=True, 
                                fill_var=sim_data_computed[f][13], 
                                fill_color='red',
                                use_legend=True, legend_loc=(.5,.0), 
                                legend_inside=False,
                                fill_label='I frame',
                                width=7.8, height=4.8, size=1.2, 
                                plot_type_left='line', 
                                plot_type_right='line',
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
            
        
        # Average latency and drop rate with I frame markings: bar
        if plot_idx == 10.45:
            plot_for_ues(ues, sim_data_computed[f][12], 
                                [sim_data_computed[f][14]],
                                [sim_data_computed[f][15]], 
                                x_axis_label='Frame index', 
                                y_axis_label=['Avg. latency [ms]', 
                                              'Drop rate [%]'],
                                linewidths=[0.6,.6,0.4], 
                                y_axis_fonts=[14,14], fill=True, 
                                fill_var=sim_data_computed[f][13], 
                                fill_color='red',
                                use_legend=True, legend_loc=(.5,.0), 
                                legend_inside=False,
                                fill_label='I frame',
                                width=7.8, height=4.8, size=1.2,
                                plot_type_left='bar', 
                                plot_type_right='bar',
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
    
    
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
            print(sim_data_computed[f][17])
            # append to a file the var above!
            with open(results_filename + '.csv', "a") as myfile:
                myfile.write(s + '\n')
            
            with open(results_filename + '_std.csv', "a") as myfile:
                myfile.write(s_std + '\n')
                
            # Also write the folder order
            with open(results_filename + '_folders.csv', "a") as myfile:
                myfile.write(stats_folder + '\n')   
            # bitrate_rank2 =[]
            # bitrate_sum = []
            # plr = []
            # for ue in range(n_ues):
            #     bitrate_sum = sim_data_trimmed[f][4][:,ue,0] + \
            #                   sim_data_trimmed[f][4][:,ue,1]
            #                   # for tti in range(0, 63980):
            #     bitrate_rank2.append((round(np.mean(bitrate_sum, axis=0),2))) 
            #     plr.append(round((100-bitrate_rank2[ue]),2))
            # print(plr)        
        # prints all detailed measurements on frame information
        if plot_idx == 10.7:
            
            print(f'Analysis of folder {f_sp.stats_path}.')
            
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
                         'Avg. Latency per frame in GoP', 
                         plot_type_left='bar',
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        # Plots avg_pck_drop_rate per frame of the GoP
        if plot_idx == 10.9:    
            plot_for_ues(ues, sim_data_computed[f][27], 
                         [sim_data_computed[f][23]], '', '%', 
                         'Avg. drop rate per frame in GoP', 
                         plot_type_left='bar',
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        # Did you know 10.1 == 10.10? So we need to jump over 10.10.
        
        # Plots the avg latency and pdr per frame of the GoP (double plot)
        if plot_idx == 10.11:
            plot_for_ues(ues, sim_data_computed[f][27], 
                                [sim_data_computed[f][22]],
                                [sim_data_computed[f][23]],
                                '', ['Latency [ms]', 'Drop Rate [%]'], 
                                'Avg. latency and drop rate per frame in GoP', 
                                plot_type_left='bar', plot_type_right='bar',
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format)
        
        # Scheduled UEs
        # Scheduled UEs: sum of co-scheduled UEs across time
        if plot_idx == 11:
            plot_for_ues([0], x_vals, [sim_data_computed[f][24]], 
                         use_legend=True, legend_inside=True, 
                         legend_loc="lower right", same_axs=True,
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
            
        # Scheduled UEs: each UE is 1 when it is scheduled and 0 when it is not 
        if plot_idx == 11.1:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][15]],
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        

        # Scheduled layers of each UE: each UE is 1 when it is scheduled and 0 when it is not 
        if plot_idx == 11.11:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][19]],
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
            
            
        # Scheduled UEs: each UE is 1 when it is scheduled and 0 when it is not 
        #                [same axis]
        if plot_idx == 11.2:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][15]], 
                         use_legend=True, legend_inside=True, 
                         legend_loc="lower right", same_axs=True,
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        
        # Count concurrent users by the bitrate
        if plot_idx == 11.3:
            plt.plot(x_vals, sim_data_computed[f][25])
            plt.xlabel(x_label_time)
            plt.ylabel('# co-scheduled UEs')
            
            if save_fig:
                plt.savefig(file_name, format=save_format)        
                print(f'Saved: {file_name}')
        
        if plot_idx == 11.4: 
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][10][:,:,l]], 
                                [sim_data_trimmed[f][15]], 
                                x_axis_label=x_label_time, 
                                y_axis_label=['Signal Power [W]', 
                                              'Scheduled UEs'],
                                savefig=save_fig, filename=file_name, 
                                saveformat=save_format) 
        
        
        # Number of co-scheduled layers per UE
        if plot_idx == 13:
            plot_for_ues(ues, x_vals, [sim_data_trimmed[f][14]],
                         x_axis_label=x_label_time, 
                         y_axis_label=['Number of layers scheduled'],
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
    
        # Packet sequences all in the same plot
        if plot_idx == 14.1:
            for ue in ues:
                pck_seq = sim_data_trimmed[f][1][ue].parent_packet_seq
                pck_seq.plot_sequence(light=True, alpha=0.6)
            plt.show()
            
            if save_fig:
                plt.savefig(file_name, format=save_format)        
                print(f'Saved: {file_name}')
            
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
                plt.savefig(file_name, format=save_format)        
                print(f'Saved: {file_name}')
        
        # Plot power of each GoB beam
        if plot_idx == 15:
            if f_sp.save_power_per_CSI_beam:
                # IDX 18 has powers of each CSI beam for each TTI for each UE.
                plot_for_ues(ues, x_vals, [sim_data_computed[f][28]], 
                             tune_opacity=False,
                             savefig=save_fig, filename=file_name, 
                             saveformat=save_format)
                    
                
        # GoB plots: plot a projection of the beams used for each ue (same ax)
        if plot_idx == 16:
            a = 2
            lims = (-a,a)
            plot_for_ues(ues, sim_data_computed[f][29], 
                         [sim_data_computed[f][30]], xlim=lims, ylim_left=lims,
                         use_legend=True, legend_inside=True, 
                         legend_loc="lower right",
                         same_axs=True, plot_type_left='scatter',
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        # GoB plots: plot a projection of the beams used for each ue (diff)
        if plot_idx == 16.1:
            a = 2
            lims = (-a,a)
            plot_for_ues(ues, sim_data_computed[f][29], 
                         [sim_data_computed[f][30]], xlim=lims, ylim_left=lims,
                         plot_type_left='scatter',
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        # GoB plots: plot a projection of all beams in the GoB [single layer]
        if plot_idx == 16.2:
            n_beams = f_sp.gob_directions.shape[-1]
            if f_sp.n_layers > 1:    
                if l == 1:    
                    idxs = [i for i in range(int(n_beams/2))]
                else:   
                    idxs = [i for i in range(int(n_beams/2), n_beams)]            
            else:
                idxs = [i for i in range(n_beams)]
       #      idxs = [  0,   4,   8,  12,  64,  68,  72,  76, 128, 132, 136, 140, 192,
       # 196, 200, 204]
            idxs = [ 34,  38,  42,  46,  98, 102, 106, 110, 162, 166, 170, 174, 226,
        230, 234, 238]
        #     idxs = [ 50,  54,  58,  62, 114, 118, 122, 126, 178, 182, 186, 190, 242,
        # 246, 250, 254]
            # pick which layer to plot the GoB
            plot_for_ues([0], sim_data_computed[f][32][idxs,0], 
                         [sim_data_computed[f][32][idxs,1]], 
                         use_legend=True, legend_inside=True, 
                         legend_loc="lower right",
                         same_axs=True, plot_type_left='scatter',
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
            
        if plot_idx == 16.3:
            n_beams = f_sp.gob_directions.shape[-1]
            if f_sp.n_layers > 1:    
                if l == 1:    
                    idxs = [i for i in range(int(n_beams/2))]
                else:   
                    idxs = [i for i in range(int(n_beams/2), n_beams)]            
            else:
                idxs = [i for i in range(n_beams)]
            
            N1 = N2 = O1 = O2 = 4
            rot_factor = 1
            gob_col_size = N2 * O2
            q = rot_factor
            
            if q <= 3:
                pass 
            elif 4 <= q <= 7:
                q = q + (gob_col_size - O2) * 1
            elif 8 <= q <= 11:
                q = q - 8 + (gob_col_size - O2) * 2
            elif 12 <= q <= 15:
                q = q - 12 + (gob_col_size - O2) * 3
            else: 
                raise Exception('That value of q is not supported. Integers 0 to 15.')
    
            # Step 2: Sum 'offsets' to get the remaining beams in the set
            q_col_idxs = q + np.arange(0,N2) * N2*O2*O1
            q_idxs_list = [q_idx + np.arange(0,N1) * N1 for q_idx in q_col_idxs]
            q_idxs = np.array(q_idxs_list).reshape((-1))
            
            if f_sp.n_layers == 2:
                q_idxs = np.hstack((q_idxs, q_idxs + int(N1) * int(N2) * int(O1) * int(O2)))
            idxs = q_idxs
            # pick which layer to plot the GoB
            plot_for_ues([0], sim_data_computed[f][32][idxs,0], 
                         [sim_data_computed[f][32][idxs,1]], 
                         use_legend=True, legend_inside=True, 
                         legend_loc="lower right",
                         same_axs=True, plot_type_left='scatter',
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)
        
        # GoB plots: plot a projection of all beams in the GoB [all ]
        if plot_idx == 16.25:
            if f_sp.n_layers == 1:
                print('Only one layer. Use 16.2 instead.')
                continue
            
            # pick which layer to plot the GoB
            
            n_beams = f_sp.gob_directions.shape[-1]
            idxs_l1 = [i for i in range(int(n_beams/2))]
            idxs_l2 = [i for i in range(int(n_beams/2), n_beams)] 
            
            plt.scatter(sim_data_computed[f][32][idxs_l1,0], 
                        sim_data_computed[f][32][idxs_l1,1], 
                        color='g', label='Layer 1')
            plt.scatter(sim_data_computed[f][32][idxs_l2,0], 
                        sim_data_computed[f][32][idxs_l2,1], 
                        color='b', label='Layer 2')
            plt.legend()
            if save_fig:
                plt.savefig(file_name, format=save_format)        
                print(f'Saved: {file_name}')
                
            plt.show()
            
        # GIFS: Plot tracks, moving beams, ...
        if 17 <= plot_idx < 18:
            # Define title to be placed in the gif
            titles = {17:    'Position and Orientation.',
                      17.01: 'Pos and Ori with beam max direction.',
                      17.02: 'Pos and Ori with beam max direction '
                             '+ constant HPWB.',
                      17.03: 'Pos and Ori with beam max direction '
                             '+ true HPWB.',
                      17.11: 'Beam max direction.', 
                      17.12: 'Beam max direction + constant HPWB. ',
                      17.13: 'Beam max direction + true HPWB.'}
            
            title = titles[plot_idx]

            colors = ['C' + str(i) for i in range(8)]

            # position is sim_data_computed[f][33]
            # orientation is sim_data_computed[f][34]

            FPS = 40
            ttis_per_sec = 4000
            time_compression_ratio = f_sp.time_compression_ratio
            samples_per_sec = ttis_per_sec / time_compression_ratio
            
            # duration of the video
            duration = int((ttis[-1] + 1) / ttis_per_sec) # 16
            
            if duration > (ttis[-1] + 1) / ttis_per_sec:
                raise Exception('Duration is longer than the simulated data!')


            # Get compressed and uncompressed time factors. Simulation 
            # variables have has many ttis as the simulation (uncompressed), 
            # but the variables directly imported from the generation, like the
            # track, have less TTIs.
            time_factor_uncompressed = ttis_per_sec / FPS
            time_factor_compressed = samples_per_sec / FPS

            timestamps = np.linspace(0, duration, FPS * duration + 1)
            
            # If tracks are needed
            if plot_idx in [17, 17.01, 17.02, 17.03]:   
                if timestamps[-1] * time_factor_compressed > \
                   sim_data_computed[f][33].shape[2]:
                    print('There are more timestamps than position samples. '
                          'Problem with compression or is the gif duration too'
                          'nlarge?')

            # If beams are needed
            if plot_idx in [17.01, 17.02, 17.03, 17.11, 17.12, 17.13]:
                # h = bs height - ue height
                bs_pos = [6,6,3]

                if timestamps[-1] * time_factor_uncompressed >= n_ttis: 
                    print('There are more timestamps than beam samples. Maybe'
                          'not enough TTIs were simulated.')
                
                # A tangent needed for some projecting beams
                tan_aux = np.tan(np.deg2rad(sim_data_computed[f][36]))                    
            
            # Start Matplot subplot
            #fig, ax = plt.subplots(projection='3d', figsize=(6,5))
            fig = plt.figure(figsize=(6,5))
            ax = fig.add_subplot(projection='3d') 

            # Setup View Angle
            # E.g. combinations for (ele,azi):
            #     a) top: (90, -90); b) side: (0, -90.1); c) 3d: (30,-120)
            view = 'top' # 'side', '3d', 'custom'
            if view == 'top':
                elev_view = 90
                azim_view = -90    
            elif view == 'side':
                elev_view = 90
                azim_view = -90
            elif view == '3d':
                elev_view = 90
                azim_view = -90
            else:
                elev_view = 34
                azim_view = 45
            ax.view_init(elev=elev_view, azim=azim_view)
        
            # room size [m]
            xlim = [1, 15]
            ylim_left = [1, 15]
            zlim = [0, 2]

            

            # method to get frames
            def make_frame(t):
                
                # clear previous frame
                ax.clear()
                
                # Get beam and track indices
                idx_uncomp = int(np.where(timestamps == t)[0][0] * 
                                 time_factor_uncompressed)
                idx_comp = int(np.where(timestamps == t)[0][0] * 
                               time_factor_compressed)
        
        
                if plot_idx in [17.01, 17.02, 17.03, 17.11, 17.12, 17.13]:
                    # if the plot includes tracks, project onto the actual 
                    # pos, otherwise project on the initial height.
                    if plot_idx in [17.01, 17.02, 17.03]:
                        h_proj = sim_data_computed[f][33][:,2, idx_comp]
                    else:
                        h_proj = f_sp.initial_pos_backup[:,2]
                    
                    h = bs_pos[2] - h_proj
                    
                    x_main_beam = bs_pos[0] + h * tan_aux[idx_uncomp,:,1]
                    y_main_beam = bs_pos[1] + h * tan_aux[idx_uncomp,:,0]

                for ue in range(n_ues):
                    if plot_idx in [17, 17.01, 17.02, 17.03]:
                        # PLOT TRACK
                        # Plot position [x y z]
                        ax.scatter(sim_data_computed[f][33][ue, 0, idx_comp],
                                   sim_data_computed[f][33][ue, 1, idx_comp],
                                   sim_data_computed[f][33][ue, 2, idx_comp],
                                   s=25, color=colors[int(ue/4)])
                        
                        # Plot orientation vector
                        ax.plot(sim_data_computed[f][34][ue, 0, idx_comp, :],
                                sim_data_computed[f][34][ue, 1, idx_comp, :],
                                sim_data_computed[f][34][ue, 2, idx_comp, :],
                                linewidth=1.5, color=colors[int(ue/4)])
                    
                    if plot_idx in [17.01, 17.02, 17.03, 17.11, 17.12, 17.13]:
                        # PLOT BEAMS! 
                        
                        # Red when not scheduled
                        if sim_data_trimmed[f][15][idx_uncomp][ue] == 0:
                            m = 'X'
                            c = 'r'
                        else:
                            m = 'o'
                            c = 'b'
                        
                        # plot beams at the UE0 height
                        ax.scatter(x_main_beam[ue], y_main_beam[ue], 
                                   h_proj[ue], marker=m, color=c)
                        
                        
                    if plot_idx in [17.02, 17.03, 17.12, 17.13]:
                        
                        beam_azi = sim_data_computed[f][36][idx_uncomp, ue, 0]
                        beam_el = sim_data_computed[f][36][idx_uncomp, ue, 1]
                        
                        # TODO: *screaming while pulling hair* 
                        #       WHY IS THIS NOT WORKING?!?!?!
                        # curr_dir = [beam_azi, beam_el]
                        # beam_idx = [i for i in range(directions.shape[1])
                        #             if np.array_equal(directions[:, i], 
                        #                               np.array(curr_dir))]
                        # a = [i for i in range(directions.shape[1])
                        #      if i == 150]
                        
                        dir_idxs_azi = \
                            np.where(f_sp.gob_directions[0,:] == beam_azi)
                        dir_idxs_el = \
                            np.where(f_sp.gob_directions[1,:] == beam_el)
                        
                        
                        beam_idx = np.intersect1d(dir_idxs_azi, dir_idxs_el)[0]
                        
                        # in [degree]
                        if plot_idx in [17.02, 17.12]:
                            azi_HPBW = el_HPBW = 25
                        else:
                            azi_HPBW = sim_data_computed[f][35][beam_idx][0] 
                            el_HPBW = sim_data_computed[f][35][beam_idx][1]
                        
                        # Note: no point in doing this in pre-processing 
                        #       because we always need a loop and this one here 
                        #       will run anyway
                        
                        # -3 dB computations for the current beam HPBWs
                        x_azi_plus = h[ue] * np.tan(np.deg2rad(beam_el))
                        y_azi_plus = h[ue] * np.tan(np.deg2rad(beam_azi + 
                                                               azi_HPBW / 2))
                        x_azi_minus = h[ue] * np.tan(np.deg2rad(beam_el))
                        y_azi_minus = h[ue] * np.tan(np.deg2rad(beam_azi - 
                                                                azi_HPBW / 2))
                        
                        x_el_plus = h[ue] * np.tan(np.deg2rad(beam_el + 
                                                              el_HPBW / 2))
                        y_el_plus = h[ue] * np.tan(np.deg2rad(beam_azi))
                        x_el_minus = h[ue] * np.tan(np.deg2rad(beam_el - 
                                                               el_HPBW / 2))
                        y_el_minus = h[ue] * np.tan(np.deg2rad(beam_azi))
                        
                        # And offset them to/from the centre.
                        x_azi_plus += bs_pos[0]
                        x_azi_minus += bs_pos[0]
                        x_el_plus += bs_pos[0]
                        x_el_minus += bs_pos[0]
                        y_azi_plus += bs_pos[1]
                        y_azi_minus += bs_pos[1]
                        y_el_plus += bs_pos[1]
                        y_el_minus += bs_pos[1]

                        # the -3dB marks
                        ax.scatter(x_azi_plus, y_azi_plus, h_proj[ue], 
                                   marker='+', color=c)
                        ax.scatter(x_azi_minus, y_azi_minus, h_proj[ue], 
                                   marker='+', color=c)
                        ax.scatter(x_el_plus, y_el_plus, h_proj[ue], 
                                   marker='+', color=c)
                        ax.scatter(x_el_minus, y_el_minus, h_proj[ue], 
                                   marker='+', color=c)
                   
                        
                # Set limits, labels and title
                ax.set_xlim(xlim)
                ax.set_ylim(ylim_left)
                ax.set_zlim(zlim)
                ax.set_xlabel('x-axis [m]')
                ax.set_ylabel('y-axis [m]')
                ax.set_zlabel('z-axis [m]')
                    
                # Take ticks away if view from the top
                if elev_view == 90:
                    ax.set_zticks([])
                if azim_view == -90.1:
                    ax.set_yticks([])
                    
                # Set plot title
                ax.set_title(title + f'Time: {t:.3f} s')

                # returning numpy image
                return mplfig_to_npimage(fig)
            
            # creating animation
            animation = VideoClip(make_frame, duration = duration) # in seconds
            
            # displaying animation with auto play and looping
            #animation.ipython_display(fps = FPS, loop = True, autoplay = True)
            
            video_path = 'Videos' + '//' + stats_folder + '//'
            
            if not ut.isdir(video_path):
                ut.makedirs(video_path)

            filename = video_path + raw_file_name + view + ".mp4"
            
            animation.write_videofile(filename, fps=FPS, audio=False, 
                                      preset='ultrafast')
            
        if plot_idx == 18.1:
            print(sim_data_computed[f][37])
           
        # GoB plots: plot a projection of all beams in the GoB
        if plot_idx == 19.1:
            plot_for_ues(ues, [sim_data_trimmed[f][2][:,:,l][:,:,:,0], 
                               sim_data_trimmed[f][2][:,:,l][:,:,:,1]], 
                         use_legend=True, legend_inside=True, 
                         legend_loc="lower right",
                         same_axs=True, plot_type_left='scatter',
                         savefig=save_fig, filename=file_name, 
                         saveformat=save_format)    
            
