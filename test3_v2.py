# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 12:52:18 2021

@author: Morais
"""
import numpy as np
import matplotlib.pyplot as plt

import utils as ut

# TODO: remove?
import matplotlib.ticker as ticker 


def plot_ues(ue_list, x_vals, y_vals, x_axis_label='', y_axis_label='',
             title='', linewidths='', tune_opacity=False, opacity=[],
             y_labels='', xlim=[], ylim=[], tight_x_axis=False,
             tight_y_axis=False, use_legend=False,
             legend_inside=False, legend_loc="center",
             ncols=1, size=1, width=6.4, 
             height=4.8, same_axs=False, n1=-1, n2=-1,
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
    n_y_vals = len(y_vals)
    
    # Adjust and compute other variables from the inputs.
    if y_vals[0] is None:
        raise Exception('y_vals only has Nones in each UE...')

    if linewidths == '':
        if same_axs:
            linewidths = [1 for ue in ue_list]
        else:
            linewidths = [1 for y_val in y_vals]
    
    # Number of traces in the same plot
    n_opacity = n_y_vals if not same_axs else n_ues*n_y_vals
        
    if opacity == []:
        if tune_opacity:
            opacity = [1.2 / n_opacity] * n_opacity
        else:
            opacity = [1] * n_opacity
            
    if len(opacity) != n_opacity:
        raise Exception('Opacity length is not the correct one.')
    
    if y_labels == '':
        if same_axs:
            y_labels = [f'UE {ue}' for ue in ue_list]
        else:
            y_labels = ['' for y_val in y_vals]
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
    for ue in ue_list:
        
        # Get ax index
        if same_axs:
            idx = 0
        else:
            if n2 > 1:
                aux = int(n_ues / 2)
                if ue < aux:
                    idx = (0, ue)
                else:
                    idx = (1, ue - aux)
            else:
                idx = ue_list.index(ue)
                
        ax1_handle = axs[idx]
        
        # Select x_data
        if x_vals.ndim == 1:
            x_data = x_vals
        else:
            x_data = x_vals[:,ue]
        
        for y_idx in range(n_y_vals):
            if same_axs:
                if n_ues == 1:
                    p_idx = y_idx
                else:
                    p_idx = ue # plot idx
            else:
                p_idx = y_idx
                ax1_handle.set_title(f'UE {ue}')
            
            # Select y_data
            if y_vals[y_idx].ndim == 1:
                y_data = y_vals[y_idx]
            else:
                y_data = y_vals[y_idx][:,ue]
            
            
            # Check sizes of x and y: trimming problem?
            if len(x_data) != len(y_data):
                raise Exception('x and y values have different dimensions. '
                                'Problem with trimming?')
            
            # Try to plot the left axis
            try:
                if plot_type_left == 'line':
                    ax1_handle.plot(x_data, y_data, alpha=opacity[p_idx], 
                                   linewidth=linewidths[p_idx], 
                                   label=y_labels[p_idx])
                elif plot_type_left== 'scatter':
                    ax1_handle.scatter(x_data, y_data)  
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


    # if it is double, set a couple of things if they are not set in advance:
    # alpha=0.6
    # color_ax1 ='g'
    # color_ax2 ='b'
    # linewidth_ax2=1

            # Try to plot the right axis
            ax2_handle = ax1_handle.twinx()
            try:
                if plot_type_right == 'line':
                    ax2_handle.plot(x_data, y_data, alpha=opacity[p_idx], 
                                   linewidth=linewidths[p_idx], 
                                   label=y_labels[p_idx])
                elif plot_type_right == 'scatter':
                    ax2_handle.scatter(x_data, y_data)  
                elif plot_type_right == 'bar':
                    ax2_handle.bar(x_data, y_data)
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
        
                
        # Set X and Y labels 
        ax1_handle.set_xlabel(x_axis_label)
        ax1_handle.set_ylabel(y_axis_label)
        
        
        # Set X and Y limits (MUST BE ONE FOR LEFT AND RIGHT!)
        if xlim != []:
            if isinstance(ylim, tuple):
                ax1_handle.set_xlim(xlim)
            elif isinstance(ylim, list) and len(xlim) == n_ues:
                ax1_handle.set_xlim(xlim[ue])
            else:
                raise Exception('xlim badly formatted: list of tuples, one '
                                'tuple for each ue')
        else:
            # ax1_handle.set_xlim([min(x_data)-1, max(x_data)+1])
            ax1_handle.autoscale(enable=True, axis='x', tight=tight_x_axis)
        
        if ylim != []:
            if isinstance(ylim, tuple):
                ax1_handle.set_ylim(ylim)
            elif isinstance(ylim, list) and len(ylim) == n_ues:
                ax1_handle.set_ylim(ylim[ue])
            else:
                raise Exception('ylim badly formatted: list of tuples, one '
                                'tuple for each ue')
        else:
            ax1_handle.autoscale(enable=True, axis='y', tight=tight_y_axis)
            
        # Set legend
        if use_legend and legend_inside:
            legend_handle = ax1_handle.legend(loc=legend_loc)
        
    if use_legend and not legend_inside:
        handles, labels = ax1_handle.get_legend_handles_labels()
        legend_handle = fig.legend(handles, labels, loc=legend_loc, 
                                   fancybox=True, shadow=True, ncol=ncols)
        
    if use_legend:
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