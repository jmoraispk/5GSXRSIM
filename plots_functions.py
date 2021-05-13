# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 14:34:43 2021

@author: Morais
"""


import numpy as np
import matplotlib.pyplot as plt

# TODO: check if really needed: see Testing file for app_trafic_plots.
import matplotlib.ticker as ticker 

import utils as ut

def plot_for_ues(ue_list, x_vals, y_vals, x_axis_label='', y_axis_label='',
                 title='', linewidths='', y_labels='', use_legend=False,
                 legend_inside=False, legend_loc="center", 
                 legend_coords=(0.53, -0.01), ncols=1, size=1, width=6.4, 
                 height=4.8, filename='', savefig=False, 
                 uniform_scale=[], same_axs=False, plot_type='line'):
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
            plt.savefig(filename + '.pdf', format='pdf')
            
        print(f'Saved: {filename}')
    
    plt.show()
    return axs


def plot_for_ues_double(ue_list, x_vals, y_vals_left, y_vals_right,
                        x_label, y_label, title='', linewidths='',
                        limits_ax1='', limits_ax2='',
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
        # ax1_handle.tick_params(axis='y', colors='g')
        # ax2_handle.tick_params(axis='y', colors='b')
        tickspacing_ax1=tickspacing_ax2=np.zeros(len(ue_list))
        if limits_ax1:
            if len(ue_list) == 1 and limits_ax1 != []:
                print(limits_ax1[0])
                print(limits_ax1[1])
                ax1_handle.set_ylim(limits_ax1[0],limits_ax1[1])
                tickspacing_ax1=(limits_ax1[1]-limits_ax1[0])/no_ticks_ax1[ue]
                ax1_handle.yaxis.set_major_locator(ticker.MultipleLocator(base=tickspacing_ax1))
                
            elif len(ue_list) > 1 and limits_ax1[ue] != []:
                ax1_handle.set_ylim(limits_ax1[ue][0],limits_ax1[ue][1])
                tickspacing_ax1[ue]=(limits_ax1[ue][1]-limits_ax1[ue][0])/no_ticks_ax1[ue]
                ax1_handle.yaxis.set_major_locator(ticker.MultipleLocator(base=tickspacing_ax1[ue]))
        
        if limits_ax2:
            if len(ue_list) == 1 and limits_ax2 != []:
               print(limits_ax2[0])
               print(limits_ax2[1])
               ax2_handle.set_ylim(limits_ax2[0],limits_ax2[1])
               tickspacing_ax2=(limits_ax2[1]-limits_ax2[0])/no_ticks_ax2[ue]
               ax2_handle.yaxis.set_major_locator(ticker.MultipleLocator(base=tickspacing_ax2))
            elif len(ue_list) > 1 and limits_ax2[ue] != []:
                 ax2_handle.set_ylim(limits_ax2[ue][0],limits_ax2[ue][1])
                 tickspacing_ax2[ue]=(limits_ax2[ue][1]-limits_ax2[ue][0])/no_ticks_ax2[ue]
                 ax2_handle.yaxis.set_major_locator(ticker.MultipleLocator(base=tickspacing_ax2[ue]))
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
            fig.savefig(filename + '.pdf', format='pdf',
                        bbox_extra_artists=(legend_handle,), 
                        bbox_inches='tight')
        else:
            plt.savefig(filename + '.pdf', format='pdf')
        
        print(f'Saved: {filename}')
        
    plt.show()
 
    

def compute_set_1(plot_idx, ues, x_vals, ttis, block_errors, n_blocks, beams,
                  buffers, sp, bitrate_realised, 
                  stats_dir_temp, first_tti_temp, last_tti_temp, 
                  stats_dir, first_tti, last_tti, instantaneous_bler, 
                  running_avg_bler, beam_formula, avg_lat, drop_rate, frames, 
                  n_frames, I_frames, running_avg_bitrate, rolling_avg_bitrate,
                  results_filename, use_in_loop):
    
    
    # initialise all variables to NULL if they haven't been computed before!
    if stats_dir != stats_dir_temp or first_tti != first_tti_temp \
                                   or last_tti != last_tti_temp:
    
        instantaneous_bler, running_avg_bler, beam_formula, \
            avg_lat, drop_rate, frames, I_frames, running_avg_bitrate, \
                rolling_avg_bitrate= tuple([None] * 9)
    
    
    ########### PART OF PRE COMPUTATIONS: data required for given plots ######
    
    idxs_require_running_bitrate = [1.1, 'all']
    
    idxs_require_rolling_avg_bitrate = [1.2, 'all']
    
    idxs_require_inst_bler = \
        [2.1, 2.2, 2.4, 5.7, 7.1, 7.2, 7.3, 7.35, 7.4, 7.5, 8, 9.1, 9.2, 'all']
    
    # These indices must be in inst.bler also
    idxs_require_running_bler = \
        [7.2, 7.3, 7.35, 7.4, 8, 'all']
    
    idxs_require_signal_power = [3.2, 'all']    
    
    idxs_require_beam_formula = \
        [5.1, 5.2, 5.3]
    
    idxs_require_latencies = \
        [10.1, 10.15, 10.2, 10.25, 10.3, 10.31, 10.4, 10.45, 10.8, 10.9, 'all']
    
    
    # Compute Running bitrate
    if plot_idx in idxs_require_running_bitrate \
        and running_avg_bitrate is None:
        running_avg_bitrate = np.zeros(bitrate_realised.shape)
        
        for ue in ues:
            for tti in range(len(ttis)):
                running_avg_bitrate[tti,ue] = (sum(bitrate_realised[0:tti,ue]) 
                                               / (tti+1))
                
                
    
    # Compute Rolling avg bitrate
    if plot_idx in idxs_require_rolling_avg_bitrate \
        and rolling_avg_bitrate is None:
        rolling_avg_bitrate = np.zeros(bitrate_realised.shape)
        
        # TTIs of 1 GoP
        rolling_interval_bitrate = int(sp.GoP/sp.FPS / sp.TTI_dur_in_secs)  
        
        for ue in ues:
            for tti in range(len(ttis)):
                if tti < rolling_interval_bitrate:
                    rolling_avg_bitrate[tti,ue] = \
                        sum(bitrate_realised[0:tti,ue]) / (tti+1)
                else:
                    rolling_avg_bitrate[tti,ue] = \
                        (sum(bitrate_realised[tti-rolling_interval_bitrate:tti,ue]) / 
                         rolling_interval_bitrate)
    
    
    if plot_idx in idxs_require_signal_power:
        pass
        # if sp.save_per_prb_variables:
        #     sig_pow = np.mean
    
    # Compute Instantaneous BLER 
    if (plot_idx in idxs_require_inst_bler) and instantaneous_bler is None:
        instantaneous_bler = np.zeros(block_errors.shape)
        for ue in ues:
            instantaneous_bler[:,ue] = \
                np.array([block_errors[i,ue] / n_blocks[i,ue] * 100
                          if n_blocks[i,ue] != 0 else 0 
                          for i in range(len(ttis))])
        
        
    # Compute Running average of BLER 
    if plot_idx in idxs_require_running_bler and running_avg_bler is None:
        running_avg_bler = np.zeros(block_errors.shape)
        for ue in ues:
            for tti in range(len(ttis)):
                running_avg_bler[tti,ue] = \
                    (running_avg_bler[tti-1,ue] * tti + 
                     instantaneous_bler[tti,ue]) / (tti+1)
        
        
    if plot_idx in idxs_require_beam_formula and beam_formula is None:
        beam_formula = beams[:,:,0] + beams[:,:,1] * 10
        
        # Further processing for filtered samples
        if plot_idx in idxs_require_beam_formula[1:]:
            # keep constant if the next value is 0
            for tti in range(len(x_vals)):
                for ue in range(len(ues)):
                    if beam_formula[tti, ue] == 0:
                        beam_formula[tti, ue] = beam_formula[tti-1, ue]
            
            # address first zeros too
            for ue in range(len(ues)):
                if beam_formula[0, ue] == 0:
                    # find next non-zero value
                    # idx = ut.first_nonzero(beam_formula, invalid_val=-1):
                    idx = (beam_formula[:, ue] != 0).argmax()
                    for tti in range(0, idx):
                        beam_formula[tti, ue] = beam_formula[idx, ue]
            
        

    if plot_idx in idxs_require_latencies and avg_lat is None:
        # Count number of periods and frames
        n_periods = round(last_tti * 0.00025 / 0.2)
        
        n_frames = n_periods * sp.GoP
        
        avg_lat = np.zeros([n_frames, len(ues)])
        drop_rate = np.zeros([n_frames, len(ues)])
        
        # Compute packet success percentages, average latencies and drop rates
        for ue in ues:
            for per in range(n_periods):
                for frm in range(sp.GoP):
                    packets_sent = \
                        buffers[ue].frame_infos[per][frm].successful_packets
                    dropped_packets = \
                        buffers[ue].frame_infos[per][frm].dropped_packets
                    total_packets = \
                        packets_sent + dropped_packets
                    
                    frame_idx = per * sp.GoP + frm
                    
                    drop_rate[frame_idx][ue] = \
                        dropped_packets / total_packets * 100
                    
                    avg_lat[frame_idx][ue] = \
                        (buffers[ue].frame_infos[per][frm].avg_lat.microseconds
                         / 1e3)
                    
        frames = np.arange(n_frames)
        # plt.plot(frames, avg_lat)
        # plt.plot(frames, drop_rate)
        
        #plot_for_ues(ues, frames, [avg_lat], y_axis_label='Frame latency [ms]')
        #plot_for_ues(ues, frames, [drop_rate], y_axis_label='Drop rate [%]')
        
        
        # Identify I frames
        I_frames = np.zeros([n_frames, len(ues)])
        I_frames[0,:] = 1
        for f_idx in range(len(frames)):
            if frames[f_idx] % 6 == 0:
                I_frames[f_idx,:] = 1
                
                
                
    if plot_idx == 10.8:
        # I frames have more packets than P frames, we need to account for that
        # when computing the overall average PDR from per frame measurements
        
        
        # I frames: 
        I_frame_idxs = [i for i in range(n_frames) if i % sp.GoP == 0]
        aux_avg_drop_rate = np.zeros(drop_rate.shape)
        aux_avg_drop_rate[:] = drop_rate[:] 
        aux_avg_drop_rate[I_frame_idxs, :] *= 1 / sp.IP_ratio
        
        # Don't try to understand this math... Basically we need to weigh the 
        # number of packets of each frame and divide by the number of frames
        # and something else to account for the scalling we did previously.
        b = (1/sp.IP_ratio) * ((sp.GoP - 1) * sp.IP_ratio + 1)
        avg_pck_drop_rate_per_frame = \
            np.sum(aux_avg_drop_rate, 0) / (n_frames * b / sp.GoP)
            
        # avg_pck_drop_rate_per_frame = \
        #     np.sum(aux_avg_drop_rate, 0) / (n_frames * 10 / 6)
            
        # avg_pck_drop_rate_per_frame2 = sum(drop_rate, 0) / n_frames
        folder = stats_dir.split('\\')[-2]
        avg_pdr = round(np.mean(avg_pck_drop_rate_per_frame),2)
        avg_pdr_std = round(np.std(avg_pck_drop_rate_per_frame),2)
        
        # repeat for latency
        # aux_avg_lat = avg_lat[I_frame_idxs, :] * 5
        # avg_pck_lat_per_frame = sum(aux_avg_lat, 0) / (n_frames * 6)
        # avg_pck_lat_per_frame2 = sum(avg_lat, 0) / n_frames
        # avg_pck_lat = round(np.mean(avg_pck_lat_per_frame),2)
        # avg_pck_lat_std = round(np.std(avg_pck_lat_per_frame),2)
        
        
        s = f'{avg_pdr}'
        s_std = f'{avg_pdr_std}'
        s_folder = f'{folder}'
        
        print('Done for folder: ' + s_folder + '. Result: ' + s)
        
        # Only write to file in case this is a loop. Otherwise, just print
        if use_in_loop:
            # append to a file the var above!
            with open(results_filename + '.csv', "a") as myfile:
                myfile.write(s + '\n')
            
            with open(results_filename + '_std.csv', "a") as myfile:
                myfile.write(s_std + '\n')
                
            # Also write the folder order
            with open(results_filename + '_folders.csv', "a") as myfile:
                myfile.write(s_folder + '\n')
        
        
    if plot_idx == 10.9:
        # METRICS PER FRAME IN THE GOP
        avg_pck_lat_per_frame_in_gop = np.zeros([sp.GoP, len(ues)])
        avg_pck_drop_rate_per_frame_in_gop = np.zeros([sp.GoP, len(ues)])
        
        
        I_frame_idxs = np.array([i for i in range(n_frames) 
                                 if i % sp.GoP == 0])
        
        for idx_in_gop in range(sp.GoP):
            idxs = I_frame_idxs + idx_in_gop
            avg_pck_lat_per_frame_in_gop[idx_in_gop,:] = \
                np.mean(avg_lat[idxs, :], 0)
            avg_pck_drop_rate_per_frame_in_gop[idx_in_gop,:] = \
                np.mean(drop_rate[idxs, :], 0)
        
        # PER I frame
        avg_pck_lat_per_I_frame = avg_pck_lat_per_frame_in_gop[0, :]
        avg_pck_drop_rate_per_I_frame = \
            avg_pck_drop_rate_per_frame_in_gop[0, :]
        
        # PER P frame
        avg_pck_lat_per_P_frame = \
            sum(avg_pck_lat_per_frame_in_gop[1:, :],0) / (sp.GoP - 1)
        avg_pck_drop_rate_per_P_frame = \
            sum(avg_pck_drop_rate_per_frame_in_gop[1:, :],0) / (sp.GoP - 1)
            
            
        # PER FRAME METRICS
        # packs proportion: if there are 5 P frames at 20% rat with I frame, 
        # then there are as many packets in the I frame as in those 5 P frames.
        # This variable guarantees weighing # pcks per frame
        P_frame_pcks_proportion = (sp.GoP - 1) * sp.IP_ratio 
        # denominator
        d = P_frame_pcks_proportion + 1
        avg_pck_lat_per_frame = \
            (avg_pck_lat_per_I_frame + avg_pck_lat_per_P_frame) / d
        avg_pck_drop_rate_per_frame = \
            (avg_pck_drop_rate_per_I_frame + avg_pck_drop_rate_per_P_frame) / d
        
        
        # Plots for frame in GoP
        gop_idxs = np.arange(0, sp.GoP)
        # plot_for_ues(ues, gop_idxs, [avg_pck_lat_per_frame_in_gop],
        #              '', 'ms', 
        #              'Avg. Latency per frame in GoP', plot_type='bar')
        
        # plot_for_ues(ues, gop_idxs, [avg_pck_drop_rate_per_frame_in_gop],
        #              '', '%', 
        #              'Avg. drop rate per frame in GoP', plot_type='bar')
        
        plot_for_ues_double(ues, gop_idxs, 
                            [avg_pck_lat_per_frame_in_gop],
                            [avg_pck_drop_rate_per_frame_in_gop],
                            '', ['Latency [ms]', 'Drop Rate [%]'], 
                            'Avg. latency and drop rate per frame in GoP', 
                            plot_type_left='bar', plot_type_right='bar')
        
        folder = stats_dir.split('\\')[-2]
        
        print(f'Analysis of folder {folder}.')
        
        # Latency stats
        avg_latency = round(np.mean(avg_pck_lat_per_frame),2)
        avg_latency_std = round(np.std(avg_pck_lat_per_frame),2)     
        print(f'Avg. Latency is {avg_latency} ms,' + \
              f' with STD of {avg_latency_std} ms.')
            
        avg_pck_lat_per_frame = \
            np.round(avg_pck_lat_per_frame, 2)
        avg_pck_lat_per_I_frame = \
            np.round(avg_pck_lat_per_I_frame, 2)
        avg_pck_lat_per_P_frame = \
            np.round(avg_pck_lat_per_P_frame, 2)
        print(f'Avg. latency per frames: {avg_pck_lat_per_frame} ms.')
        print(f'Avg. latency for I frames: {avg_pck_lat_per_I_frame} ms.')
        print(f'Avg. latency for P frames: {avg_pck_lat_per_P_frame} ms.')
        
        # Drop rate stats
        avg_pdr = round(np.mean(avg_pck_drop_rate_per_frame),2)
        avg_pdr_std = round(np.std(avg_pck_drop_rate_per_frame),2)
        print(f'Avg. PDR is {avg_pdr} %,' + \
              f' with STD of {avg_pdr_std} %.')
        
        avg_pck_drop_rate_per_frame = \
            np.round(avg_pck_drop_rate_per_frame, 2)
        avg_pck_drop_rate_per_I_frame = \
            np.round(avg_pck_drop_rate_per_I_frame, 2)
        avg_pck_drop_rate_per_P_frame = \
            np.round(avg_pck_drop_rate_per_P_frame, 2)
        print(f'Avg. drop rate per frames: {avg_pck_drop_rate_per_frame} %.')
        print(f'Avg. drop rate for I frames: {avg_pck_drop_rate_per_I_frame} %.')
        print(f'Avg. drop rate for P frames: {avg_pck_drop_rate_per_P_frame} %.')
        
    
    return (instantaneous_bler, running_avg_bler, beam_formula, \
            avg_lat, drop_rate, frames, n_frames, I_frames, \
            running_avg_bitrate, rolling_avg_bitrate)
    
        
    
def plot_set_1(plot_idx, save_fig, ues, ttis, x_vals, x_vals_label, sp, 
               buffers, stats_dir, x_vals_save_suffix, bitrate_realised, 
               signal_power, signal_power_prb, sinr_estimated, sinr_realised, 
               olla_param, dl_interference, mcs, beams, scheduled_UEs, channel, 
               su_mimo_setting, beam_formula, instantaneous_bler, 
               running_avg_bler, avg_lat, drop_rate, frames, I_frames,
               running_avg_bitrate, rolling_avg_bitrate, dl_interference_est):
    
        
    """
        THE MEANING OF EACH PLOT INDEX IS IN PLOTS_PHASE1.PY, for now...
    """  
    # TODO: move this list to sls_plot... or find a way of having 
    #       all lists synched across functions, and in one place.
    plots_to_skip = [10.8, 10.9]
    
    # Skip the indices that are for computations only.
    if plot_idx in plots_to_skip:
        return
    
    
    # File naming
    freq_save_suffix = '_' + str(round(sp.freq / 1e9,1)) + 'ghz'
    fname_preffix = ut.get_cwd() + '\\Plots\\' + \
                    stats_dir.split('\\')[-2] + '\\'
    fname_suffix = x_vals_save_suffix + freq_save_suffix
    
    if not ut.isdir(fname_preffix):
        ut.makedirs(fname_preffix)
    
    def fname(name):
        return fname_preffix + name + fname_suffix

    
    ########################### PART OF PLOTING ##############################
    
    # Channel across time for many PRBs
    # channel across PRBs for one time (tti)
    if plot_idx == 0.1 or plot_idx == 'all':    
        plot_for_ues(ues, x_vals, [channel],
                     x_axis_label=x_vals_label, y_axis_label='Power [dBW]',
                     filename=fname('channel'), savefig=save_fig)
    
    
    if plot_idx == 0.2 or plot_idx == 'all':
        # TODO with variable channel_per_prb
        #  requires initialising the variable properly and changing update_channel
        #  function to compute it (can have bugs.)
        pass
    
    
    # Instantaneous Realised Bitrate
    if plot_idx == 1 or plot_idx == 'all':
        plot_for_ues(ues, x_vals, [bitrate_realised], 
                      x_vals_label, 'Mbps', 'Realised bitrate')
    
    
    # Inst. vs Running avg bitrate 
    if plot_idx == 1.1 or plot_idx == 'all':
            plot_for_ues(ues, x_vals, [bitrate_realised, running_avg_bitrate], 
                      x_vals_label, 'Mbps', 'Realised bitrate', 
                      savefig=save_fig)


    # Inst. vs Rolling avg bitrate 
    if plot_idx == 1.2 or plot_idx == 'all':
        # Make constant line
        marker_line = np.ones(bitrate_realised.shape) * 80
    
        if len(ues) == 1 and sp.avg_bitrate_dl < 400:
            # (Single-user case - normal bitrate)
            plot_for_ues(ues, x_vals, [bitrate_realised, rolling_avg_bitrate], 
                         x_axis_label=x_vals_label, 
                         y_axis_label='Bit rate [Mbps]', 
                         title='', linewidths=[0.8, 2], 
                         y_labels=['Instantaneous', 
                                   'Rolling avg.\nover GoP duration'], 
                         use_legend=True, legend_inside=True, 
                         legend_loc=(0.64,0.2), ncols=1, size=1, 
                         filename=fname('su_throughput'), savefig=save_fig, 
                         same_axs=True) 
        
        if len(ues) == 1 and sp.avg_bitrate_dl > 400:
            # (Single-user case - full buffer)
            plot_for_ues(ues, x_vals, [bitrate_realised, rolling_avg_bitrate], 
                         x_axis_label=x_vals_label, 
                         y_axis_label='Bit rate [Mbps]', 
                         title='', linewidths=[0.8, 2], 
                         y_labels=['Instantaneous', 
                                   'Rolling avg.\nover GoP duration'], 
                         use_legend=True, legend_inside=True, 
                         legend_loc=(0.645,.2), size=1, width=6.4, height=4, 
                         filename=fname('su_throughput_full'),
                         savefig=save_fig, same_axs=True) 
        
        if len(ues) > 1:
            # (Multi-user case)
            plot_for_ues(ues, x_vals, [bitrate_realised, rolling_avg_bitrate, 
                                       marker_line], 
                         x_axis_label=x_vals_label, 
                         y_axis_label='Bit rate [Mbps]', 
                         title='', linewidths=[0.3, 1.5, 1], 
                         y_labels=['Instantaneous', 
                                   'Rolling avg. over GoP duration', 
                                   'Application Bit rate'], 
                         use_legend=True, ncols=3,
                         size=1.3, filename=fname('mu_throughput'), 
                         savefig=save_fig, uniform_scale=[-8, 240]) 
        
            
    # SINR (Multi-user case)
    if plot_idx == 2 or plot_idx == 'all':
        plot_for_ues(ues, x_vals, [sinr_estimated, sinr_realised], 
                     x_axis_label=x_vals_label, y_axis_label='SINR [dB]', 
                     y_labels=['Estimated', 'Experienced'], use_legend=True, 
                     ncols=2, size=1.3,filename=fname('sinr'), 
                     savefig=save_fig)

    # SINR: only when there are active transmissions (single_user)
    if plot_idx in [2.1, 'all']:
        plot_for_ues_double([1], x_vals, [sinr_realised],
                                        [instantaneous_bler], x_vals_label, 
                                        ['Experienced SINR [dB]', 'BLER [%]'],
                                        linewidths=[1,0.4,0.15], 
                                        limits_ax1=[15,22],
                                        label_fonts=[17,17], fill=True, 
                                        fill_var=bitrate_realised, 
                                        use_legend=True,
                                        legend_loc=(1.02,.955), 
                                        legend_inside=False,
                                        fill_label='Active\ntransmissions',
                                        width=7.8, height=4.8, size=1.2,
                                        filename=fname('sinr_active_single'), 
                                        savefig=save_fig)
        
     # SINR: only when there are active transmissions (multi-user)
    if plot_idx in [2.2, 'all']:
        plot_for_ues_double(ues, x_vals, [sinr_realised],
                                        [olla_param], x_vals_label, 
                                        ['Experienced SINR [dB]', '$\Delta_{OLLA}$'],
                                        linewidths=[1,1,0.15], 
                                        label_fonts=[13,16], fill=True, 
                                        fill_var=bitrate_realised, 
                                        use_legend=True,
                                        legend_loc=(1.02,.955), 
                                        legend_inside=False,
                                        fill_label='Active\ntransmissions',
                                        width=7.8, height=4.8, size=1.2,
                                        filename=fname('sinr_active_multi'), 
                                        savefig=save_fig)

    if plot_idx in [2.3, 'all']:
        sinr_diff = sinr_realised - sinr_estimated
        plot_for_ues(ues, x_vals, [sinr_diff], x_vals_label, '[dB]', 
                     'SINR diff', filename=fname('sinr_diff'), 
                     savefig=save_fig)
        
    
    if plot_idx in [2.4, 'all']:
        sinr_diff = sinr_realised - sinr_estimated
        plot_for_ues_double(ues, x_vals, [sinr_diff], [instantaneous_bler],
                            x_vals_label, ['SINR diff [dB]', 'BLER [%]'],
                            filename=fname('sinrdiff-vs-bler'), 
                            savefig=save_fig)
        
        
    # Signal power variation across PRBs
    if plot_idx == 3 or plot_idx == 'all':
        if not sp.save_per_prb_variables:
            return
        
        if sp.n_prb > 1:
            prb_bandwidth = sp.bandwidth / sp.n_prb
            freqs_vec = (sp.freq - sp.bandwidth/2 + 
                         np.arange(0,sp.n_prb * sp.freq_compression_ratio) * 
                         prb_bandwidth)
        else:
            freqs_vec = [sp.freq]
        
        # antenna index: 
        a_idx = 3
        
        if len(freqs_vec) > 1:
            plot_for_ues(ues, freqs_vec, [signal_power_prb[a_idx,:,:].T], 
                         'Frequency', 'Watt', 
                         'Signal power variation across frequency',
                         savefig=save_fig)
        else:
            plot_for_ues(ues, freqs_vec, [signal_power_prb[a_idx,:,:].T], 
                         'Frequency', 'Watt', 
                         'Signal power variation across frequency',
                         savefig=save_fig, plot_type='scatter')
            
    
    # Signal power variation across PRBs in dB
    if plot_idx == 3.1 or plot_idx == 'all':
        if not sp.save_per_prb_variables:
            return
        
        prb_bandwidth = sp.bandwidth / sp.n_prb
        freqs_vec = (sp.freq - sp.bandwidth/2 + 
                     np.arange(0,sp.n_prb * sp.freq_compression_ratio) * 
                     prb_bandwidth)
        
        
        middle_freq = round(len(signal_power_prb.shape[-1])/2) 
        signal_power_prb_db = \
            10 * np.log10(signal_power_prb[3,:,:].T / 
                          signal_power_prb[3,:,middle_freq].T)
        
        
        plot_for_ues(ues, freqs_vec, [signal_power_prb_db], 
                     'Frequency', 'dB', 
                     'Signal power variation across frequency',
                     savefig=save_fig)
        
    # Signal power 
    if plot_idx in [3.2, 'all']:
        # Plot signal power variation across time
        plot_for_ues(ues, x_vals, [signal_power], 
                      x_vals_label, 'Watt', 'signal power across time')

    # Signal power vs interference         
    if plot_idx in [3.3, 'all']:        
        plot_for_ues(ues, x_vals, [signal_power, dl_interference], 
                      x_vals_label, '[W]', 'Signal Power vs Interference')
        
    # Signal power vs interference (dBw) [double axis]
    if plot_idx in [3.4, 'all']:      
        # compute in dBw
        dl_interference_dbw = 10 * np.log10(dl_interference[:,ues])
        signal_power_dbw = 10 * np.log10(signal_power)
        plot_for_ues_double(ues, x_vals, [signal_power_dbw], 
                            [dl_interference_dbw], x_vals_label, 
                            ['Sig. Power [dBw]', 'Int. Power [dBw]'])
    
    # Signal power vs interference (dBw) [single axis]
    if plot_idx in [3.45, 'all']:      
        # compute in dBw
        dl_interference_dbw = 10 * np.log10(dl_interference[:,ues])
        signal_power_dbw = 10 * np.log10(signal_power)
        plot_for_ues(ues, x_vals, [signal_power_dbw, dl_interference_dbw], 
                     x_vals_label, y_axis_label='[dBw]', 
                     y_labels=['Signal', 'Interference'], 
                     filename=fname('sig vs interference [dbw]'), 
                     savefig=save_fig)
            
    # Signal power vs interference (dBm)
    if plot_idx in [3.5, 'all']:
        plot_for_ues_double(ues, x_vals, 
                            [signal_power_dbw + 30, dl_interference_dbw + 30], 
                            x_vals_label, 
                            ['Signal Power [W]', 'Interference Power [W]'])

    if plot_idx in [3.6, 'all']:
        dl_interference_dbw = 10 * np.log10(dl_interference[:,ues])
        signal_power_dbw = 10 * np.log10(signal_power)
        plot_for_ues(ues, x_vals, [signal_power_dbw, dl_interference_dbw], 
                     x_vals_label, 'Power [dBW]',
                     y_labels=['Signal', 'Interference'], use_legend=True,
                     ncols=2, size=1.3, width=6.4, height=4, 
                     filename=fname('pow_vs_inter'), savefig=save_fig)
    
    # Estimated vs Realised interference
    if plot_idx in [3.7, 'all']:
        plot_for_ues(ues, x_vals, [dl_interference_est, dl_interference], 
                      x_vals_label, '[W]', 'Estimated vs real interference')
                            
    
    # Estimated vs Realised interference [dB]
    if plot_idx in [3.8, 'all']:
        plot_for_ues(ues, x_vals, [dl_interference_est, dl_interference], 
                      x_vals_label, '[W]', 'Estimated vs real interference')
                          
        
    # MCS same axs
    if plot_idx in [4.1, 'all']:
        plot_for_ues(ues, x_vals, [mcs], x_vals_label, 'MCS index',  
                     use_legend=True, legend_inside=True, 
                     legend_loc="lower right",
                     ncols=1, size=1.3, filename=fname('mcs'), 
                     savefig=save_fig, 
                     uniform_scale = [0.5, 15.5], same_axs=True)
        
    # MCS diff axs
    if plot_idx in [4.2, 'all']:
        plot_for_ues(ues, x_vals, [mcs], x_vals_label, 'MCS index', 
                     linewidths=[.4,.4,.4,.4], 
                     y_labels=['UE 0','UE 1','UE 2','UE 3'], 
                     ncols=1, size=1.3, filename=fname('mcs_separated'), 
                     savefig=save_fig, uniform_scale = [6.5, 15.5], 
                     same_axs=False)

    # MCS and instantaneous bitrate per UE
    if plot_idx in [4.3, 'all']:
        plot_for_ues_double(ues, x_vals, [mcs], [bitrate_realised], 
                            x_vals_label, 
                            y_label=['MCS index', 'Bit rate [Mbps]'],
                            savefig=save_fig)
    
    # Beams: best beam per user (filtered: prevents going back to zero when
    #        the UE is no scheduled. One plot per UE    
    if plot_idx in [5.1, 'all']:
        plot_for_ues(ues, x_vals, [beam_formula], 
                     title='Formula: azi + ele x 10')
    
    # Beams filtered: doublePlot per UE for azi and elevation values.
    if plot_idx in [5.2, 'all']:
        plot_for_ues_double(ues, x_vals, [beams[:,:,0]], [beams[:,:,1]],
                            x_vals_label, 
                            y_label=['Azimuth [º]', 'Elevation[º]'], 
                            savefig=save_fig)

    # Beam sum: used to notice beam switching easily
    if plot_idx in [5.3, 'all']:
        beam_sum = beams[:,:,0] + beams[:,:,1]
        plot_for_ues(ues, x_vals, [beam_sum],
                     x_vals_label, 'Azimuth + Elevation [º]', 
                     savefig=save_fig)
    
    # Beam sum: used to notice beam switching easily vs SINR
    if plot_idx in [5.4, 'all']:
        beam_sum = beams[:,:,0] + beams[:,:,1]
        plot_for_ues_double(ues, x_vals, [beam_sum], [sinr_realised], 
                            x_vals_label,
                            ['Azi. + El. [º]', 'SINR [dB]'],
                            savefig=save_fig)
    
    # Beam sum vs BLER
    if plot_idx in [5.5, 'all']:
        beam_sum = beams[:,:,0] + beams[:,:,1]
        plot_for_ues_double(ues, x_vals, [beam_sum], 
                            [instantaneous_bler], x_vals_label,
                            ['Azi. + El. [º]', 'BLER [%]'],
                            savefig=save_fig)
    
        
    # BLER: Instantaneous
    if plot_idx in [7.1, 'all']:
        plot_for_ues(ues, x_vals, [instantaneous_bler],
                     x_vals_label, '%', '% of Blocks with errors',
                     savefig=save_fig)
    
    # BLER: Running Average
    if plot_idx in [7.2, 'all']:       
        plot_for_ues(ues, x_vals, [running_avg_bler], 
                     x_vals_label, 'Avg. BLER [%]',
                     width=6.4, height=4.8, size=1.3, 
                     filename=fname('running_bler'), 
                     same_axs=False, savefig=save_fig)
    
    # BLER: Instantaneous + Running Average
    if plot_idx in [7.3, 'all']:
        plot_for_ues(ues, x_vals, [instantaneous_bler, running_avg_bler],
                     x_vals_label, '%', 'Running average of BLER')

    if plot_idx in [7.35, 'all']:
        plot_for_ues_double(ues, x_vals, [instantaneous_bler], 
                            [running_avg_bler],
                            x_vals_label, 
                            ['Inst. BLER [%]', 'Run. Avg. BLER [%]'], 
                            savefig=save_fig)
        
                
        
    # BLER: instantaneous BLER and realised bitrate
    if plot_idx in [7.4, 'all']:
        plot_for_ues_double(ues, x_vals, [bitrate_realised], [instantaneous_bler],
                            x_vals_label, ['Inst. bitrate [Mbps]', 'BLER [%]'],
                            filename=fname('inst._bitrate_vs_bler'),
                            savefig=save_fig)
        
        
    # BLER: Instantaneous vs realised SINR
    if plot_idx in [7.5, 'all']:
        plot_for_ues_double(ues, x_vals, [instantaneous_bler], 
                            [sinr_realised], 
                            x_vals_label,
                            ['BLER [%]', 'SINR [dB]'],
                            savefig=save_fig)
        
    # OLLA (single-user)    
    if plot_idx in [9.1, 'all']:
        # ONE UE ONLY:
        plot_for_ues_double([2], x_vals, [instantaneous_bler],
                                        [olla_param], x_vals_label, 
                                        ['BLER [%]', '$\Delta_{OLLA}$'],
                                        linewidths=[0.2,1,0.15], 
                                        label_fonts=[13,16], fill=True, 
                                        fill_var=bitrate_realised, 
                                        use_legend=True,
                                        legend_loc=(1.02,.955), 
                                        legend_inside=False,
                                        fill_label='Active\ntransmissions',
                                        width=7.8, height=4.8, size=1.2,
                                        filename=fname('olla'), 
                                        savefig=save_fig)
        
    # OLLA (multi-user)    
    if plot_idx in [9.2, 'all']:
        # MULTIPLE UEs
        plot_for_ues_double(ues, x_vals, [instantaneous_bler],
                                        [olla_param], x_vals_label, 
                                        ['BLER [%]', '$\Delta_{OLLA}$'],
                                        linewidths=[0.2,1,0.15], fill=True, 
                                        fill_var=bitrate_realised,
                                        width=7.8, height=4.8, size=1.2,
                                        filename=fname('olla'), 
                                        savefig=save_fig)
    
    # OLLA: MCS vs olla
    if plot_idx in [9.3, 'all']:
        plot_for_ues_double(ues, x_vals, [mcs], [olla_param], x_vals_label, 
                            ['CQI IDX', '$\Delta_{OLLA}$'], 'MCS and OLLA', 
                            [0.2,0.9])
     
    # OLLA: inst. bler vs olla        
    if plot_idx in [9.4, 'all']:
        plot_for_ues_double([0, 2], x_vals, [instantaneous_bler], [olla_param], 
                            x_vals_label, ['BLER [%]', '$\Delta_{OLLA}$'], 
                            'MCS and OLLA', [0.2,0.9])
    
   
    
    # LATENCY and DROP RATE
    # avg latency across frames 
    if plot_idx == 10.1 or plot_idx == 'all':
        plot_for_ues(ues, frames, [avg_lat],
                     'Frame index', 
                     'Avg. latency [ms]', '', [0.7,0.6],
                     filename=fname('latency_only'), 
                     savefig=save_fig)
    
    # avg latency across frames (bar chart)
    if plot_idx == 10.15 or plot_idx == 'all':
        plot_for_ues(ues, frames, [avg_lat],
                     'Frame index', 
                     'Avg. latency [ms]', '', [0.7,0.6],
                     filename=fname('latency_only_bar'), 
                     savefig=save_fig, plot_type='bar')
    
    # avg drop rate across frames
    if plot_idx == 10.2 or plot_idx == 'all':
        plot_for_ues(ues, frames, [drop_rate],  
                            'Frame index', 
                            'Drop rate [%]', '', [0.7,0.6],
                            filename=fname('droprate_only'), 
                            savefig=save_fig)
    
    # avg drop rate across frames (bar chart)
    if plot_idx == 10.25 or plot_idx == 'all':
        plot_for_ues(ues, frames, [drop_rate],  
                            'Frame index', 
                            'Drop rate [%]', '', [0.7,0.6],
                            filename=fname('droprate_only_bar'), 
                            savefig=save_fig, plot_type='bar')
    
    # avg latency vs drop rate across frames (no I vs P distinction)
    if plot_idx == 10.3 or plot_idx == 'all':    
        plot_for_ues_double(ues, frames, [avg_lat], [drop_rate], 
                            'Frame index', 
                            ['Avg. latency [ms]', 'Drop rate [%]'], '', 
                            [0.7,0.6], filename=fname('latency_and_droprate'), 
                            savefig=save_fig)
    
    # Same as 10.3 but showing the ticks and limits.
    if plot_idx == 10.31 or plot_idx == 'all':    
        plot_for_ues_double(ues, frames, [avg_lat], [drop_rate], 
                            'Frame index', 
                            ['Avg. latency [ms]', 'Drop rate [%]'], '', 
                            [0.7,0.6], filename=fname('latency_and_droprate'), 
                            savefig=save_fig,
                            limits_ax1=[[0,5],[0,0.6],[0,5],[0,0.6]],
                            limits_ax2=[[-0.05,0.05],[-0.05,0.05],
                                        [-0.05,0.05],[-0.05,0.05]],
                            no_ticks_ax1=[4,4,4,4],no_ticks_ax2=[4,4,4,4])
    
    # Average latency and drop rate with I frame markings: line
    if plot_idx == 10.4 or plot_idx == 'all':
        plot_for_ues_double(ues, frames, [avg_lat],
                                        [drop_rate], 'Frame index', 
                                        ['Avg. latency [ms]', 'Drop rate [%]'],
                                        linewidths=[0.6,.6,0.4], 
                                        label_fonts=[14,14], fill=True, 
                                        fill_var=I_frames, fill_color='red',
                                        use_legend=True, legend_loc=(.5,.0), 
                                        legend_inside=False,
                                        fill_label='I frame',
                                        width=7.8, height=4.8, size=1.2,
                                        filename=fname('lat_drop_rate_with_I'),
                                        savefig=save_fig, plot_type_left='line', 
                                        plot_type_right='line')
        
    
    # Average latency and drop rate with I frame markings: bar
    if plot_idx == 10.45 or plot_idx == 'all':
        plot_for_ues_double(ues, frames, [avg_lat],
                                        [drop_rate], 'Frame index', 
                                        ['Avg. latency [ms]', 'Drop rate [%]'],
                                        linewidths=[0.6,.6,0.4], 
                                        label_fonts=[14,14], fill=True, 
                                        fill_var=I_frames, fill_color='red',
                                        use_legend=True, legend_loc=(.5,.0), 
                                        legend_inside=False,
                                        fill_label='I frame',
                                        width=7.8, height=4.8, size=1.2,
                                        filename=fname('lat_drop_rate_bar'),
                                        savefig=save_fig, plot_type_left='bar', 
                                        plot_type_right='bar')

    # Scheduled UEs
    # Scheduled UEs: sum of co-scheduled UEs across time
    if plot_idx == 11 or plot_idx == 'all':
        
        count_scheduled_UEs = np.sum(scheduled_UEs, 1)
        
        # make it 2D: #ttis x 1
        count_scheduled_UEs = np.reshape(count_scheduled_UEs, 
                                         (count_scheduled_UEs.shape[0], 1))
        
        plot_for_ues([0], x_vals, [count_scheduled_UEs], same_axs=True)
        
        
    
    # Scheduled UEs: each UE is 1 when it is scheduled and 0 when it is not 
    if plot_idx in [11.1, 'all']:
        plot_for_ues(ues, x_vals, [scheduled_UEs], filename=fname('11.2'))
                
        
    if plot_idx in [11.2, 'all']:
        plot_for_ues(ues, x_vals, [scheduled_UEs], filename=fname('11.2'), 
                     same_axs=True)
    
    
    # Count concurrent users by the bitrate
    if plot_idx in [11.3, 'all']:
        bitrate_realised2 = np.nan_to_num(bitrate_realised / bitrate_realised)
        count_scheduled_UEs = np.sum(bitrate_realised2, 1)
        
        plt.plot(x_vals, count_scheduled_UEs)
        plt.xlabel(x_vals_label)
        plt.ylabel('# co-scheduled UEs')
        
    # Number of co-scheduled layers per UE
    if plot_idx == 13 or plot_idx == 'all':
        plot_for_ues(ues, x_vals, [su_mimo_setting],
                     x_vals_label, '# layers', 'Number of layers per UE')

    # Packet sequences all in the same plot
    if plot_idx == 14.1 or plot_idx == 'all':
        for ue in ues:
            buffers[ue].parent_packet_seq.plot_sequence(light=True, alpha=0.6)
        plt.show()
        
        if save_fig:
            plt.savefig('packet_sequences_same_axes.pdf', format='pdf')
    
        
    # Packet sequences all in different plots
    if plot_idx == 14.2 or plot_idx == 'all':
        n_ue = len(ues)
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
                aux = int(len(ues) / 2)
                if ue < aux:
                    idx = (0, ue)
                else:
                    idx = (1, ue - aux)
            else:
                idx = ues.index(ue)
                    
            ax_handle = axs[idx]
        
            ax_handle.plot()
            plt.sca(ax_handle)
            buffers[ue].parent_packet_seq.plot_sequence(light=True, alpha=1)
            
            ax_handle.set_title(f'UE {ue}')
            ax_handle.set_xlabel(x_vals_label)
            #ax_handle.set_ylabel('Packets per ms')
            
        
        fig.suptitle('Packets per ms')
        plt.show()
        if save_fig:
            plt.savefig('packet_sequences.pdf', format='pdf')
        
        


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



def load_sim_results(files, vars_to_load, trim_ttis):
    """
    Loads the variables from each file and prepares the data.
    """    
    n_variables = len(vars_to_load)
    n_traces = len(files)
    
    
    # sim_data is [variable_idx, trace_idx]
    sim_data = ut.make_py_list(2, [n_variables, n_traces])
    
    # Load results from simulation (stats) files
    for f in range(n_traces):
        file_path = files[f]
        
        for v_idx in range(n_variables):
            v = vars_to_load[v_idx]
            sim_data[v_idx][f] = ut.load_var_pickle(v, file_path)
        

    vars_not_to_trim = ['sp', 'buffers', 'packet_sequence_DL']
    
    # Within the variables to trim, which do not have polarisations:
    vars_with_no_pol = ['realised_bitrate_total', 'experienced_signal_power',
                        'olla', 'channel', 'scheduled_UEs']

    # Convert to NumPy arrays and trim to obtain only the useful parts
    for f in range(n_traces):
        for v_idx in range(n_variables):
            v = vars_to_load[v_idx]
            
            if v in vars_not_to_trim:
                continue
            
            # print(v)
            
            sim_data[v_idx][f] = \
                    np.array(sim_data[v_idx][f])[trim_ttis[0]:trim_ttis[1]]
            
            if v in vars_with_no_pol:
                continue
            
            # Select the layer we want (single-layer plot)
            sim_data[v_idx][f] = sim_data[v_idx][f][:,:,2]
            
            # # Select the downlink ttis only
            #sim_data[v_idx][f] = np.delete(sim_data[v_idx, f], ul_ttis, axis=0)
    
    # UL tti formula, if first and last tti are multiples of 5.
    # ul_ttis = np.arange(0, last_tti - first_tti, 5) - 1
    # ul_ttis = ul_ttis[1:]
    

    return sim_data



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
# ENABLE SUBTITLES!!!!
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


def make_filename(time_str, var, files, plots_dir):
    """
    Create the file name based on the simulation files and plotting variable 
    and plot path.
    """
    file_str = '-'

    for f in files:
        file_str += f.split('\\')[-2] + '+'
        
    return plots_dir + var + file_str[:-1]
