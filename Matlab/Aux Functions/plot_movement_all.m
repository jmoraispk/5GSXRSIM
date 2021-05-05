function [] = plot_movement_all(layout, other_participants, ...
                lim, N_usrs, update_rate, snapshot_interval, pause_interval, ...
                pause_duration, shade, view_mode, plot_mode, ...
                save_plot, make_gif, display_orientation, zoom_factor)
    %Plots the time evolution of the position and orientation of each user
    %PLOT MOVEMENT 2 ALSO PLOTS OTHER PARTICIPANTS
    
    
    %plot choice
    
    n_usr = N_usrs - size(other_participants,2);
    
    if plot_mode == 0 %plot only movement
        plot_movement2(layout, lim, n_usr, update_rate, ...
                    snapshot_interval, pause_interval, pause_duration, ...
                    shade, view_mode, make_gif, display_orientation);
    elseif plot_mode == 1 %plot movement on layout with shade
        plot_movement_on_layout(layout, lim, n_usr, update_rate, ...
                                snapshot_interval, pause_interval, ...
                                pause_duration, view_mode, ...
                                make_gif, display_orientation);
    elseif plot_mode == 2 %plot movement with all users in it
        plot_movement_plus_non_participants(layout, other_participants, ...
            lim, N_usrs, update_rate, snapshot_interval,...
            pause_interval, pause_duration, shade, view_mode, ...
            make_gif, display_orientation);
    elseif plot_mode == 3
        plot_all_not_on_layout(layout, other_participants, ...
            lim, n_usr, update_rate, snapshot_interval, ...
            pause_interval, pause_duration, shade, view_mode, ...
            make_gif, display_orientation, zoom_factor);
    end
    
    
    
    if plot_mode ~= 4
        return %job done here.
    end
    
    
    %PLOT_MODE = 4 - plot movement with every moving RX and BS
    plot_all_on_layout(layout, lim, N_usrs, other_participants, ...
                       update_rate, snapshot_interval, pause_interval, ...
                       pause_duration, view_mode, make_gif, ...
                       display_orientation);
    
    % Save plot
    
    if ~strcmp(save_plot, '')
        name = ['PlotMode', num2str(plot_mode), '_', get_time_str()];
        if strcmp(save_plot, 'fig')
            savefig([name, '.fig']);
        else
            saveas(gcf, name, save_plot);
        end
    end
end

