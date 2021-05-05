function [] = plot_movement_plus_non_participants(layout, other_participants,...
                lim, N_usrs, update_rate, snapshot_interval, pause_interval, ...
                pause_duration, shade, view_mode, make_gif, display_orientation)
    %Plots the time evolution of the position and orientation of each user
    %PLOT MOVEMENT 2 ALSO PLOTS OTHER PARTICIPANTS
    
    last_snapshot = calc_max_snap(layout.rx_track);
    
    pause_snap_interval = round(pause_interval/update_rate);
    
    %plot each position and each orientation for each snapshot
    tracks = layout.rx_track;
    n_usr = N_usrs - size(other_participants,2);
    
    [az, el] = get_view(view_mode);
                      %users, coordinate, snapshot
    all_pos = zeros([n_usr, 3, last_snapshot]);
    all_ori = zeros([n_usr, 3, last_snapshot]);
    for i=1:n_usr
        all_pos(i,1,:) = tracks(i).initial_position(1) + ...
                         tracks(i).positions(1,1:last_snapshot);
        all_pos(i,2,:) = tracks(i).initial_position(2) + ...
                         tracks(i).positions(2,1:last_snapshot);
        all_pos(i,3,:) = tracks(i).initial_position(3) + ...
                         tracks(i).positions(3,1:last_snapshot);
                     
        all_ori(i,2,:) = tracks(i).orientation(2,1:last_snapshot);
        all_ori(i,3,:) = tracks(i).orientation(3,1:last_snapshot);
    end
    
    r = 0.5; %length of the vectors (orientation)
    colors = ['k', 'b', 'y', 'm', 'c', 'r', 'g'];
    if shade
        w = 1;
    else
        w = 2;
    end
    f = figure;
    if make_gif
        axis tight manual % this ensures that getframe() returns a consistent size
        filename = ['PlotMode1_', get_time_str(), '.gif'];
    end
    
    for s=1:snapshot_interval:last_snapshot
        % Positions
        cur_pos = [all_pos(:,1,s), all_pos(:,2,s), all_pos(:,3,s)];
        scatter3(cur_pos(:,1), cur_pos(:,2), cur_pos(:,3), 'Filled');
        hold on;
        scatter3(other_participants(1,:), other_participants(2,:), ...
                 other_participants(3,:), 'k','s', 'Filled');
             
        % Orientations
        if display_orientation
            for u=1:n_usr
                theta = all_ori(u,2,s)*-1 + pi/2; 
                phi = all_ori(u,3,s); 
                target = [cur_pos(u,1) + r*sin(theta)*cos(phi), ...
                          cur_pos(u,2) + r*sin(theta)*sin(phi), ...
                          cur_pos(u,3) + r*cos(theta)];
                line([cur_pos(u,1) target(1)], [cur_pos(u,2) target(2)], ...
                     [cur_pos(u,3) target(3)], 'Color', colors(u), 'LineWidth', w);
            end
        end
        
        axis(lim);grid on;
        title(['Time: ', num2str(s*update_rate), ' s']);
        if shade == 0
            hold off;
        end
        view([az el]);
        drawnow;
        if make_gif
            % Capture the plot as an image 
            frame = getframe(f); 
            im = frame2im(frame); 
            [imind,cm] = rgb2ind(im,256); 
            % Write to the GIF File 
            if s == 1 
              imwrite(imind,cm,filename,'gif', 'Loopcount',inf, ...
                                               'DelayTime', 0.05); 
            else 
              imwrite(imind,cm,filename,'gif','WriteMode','append', ...
                                               'DelayTime', 0.05); 
            end
        end
        
        curr_divisor = 0;
        if (s/pause_snap_interval) > curr_divisor
            curr_divisor = round(s/pause_snap_interval); %#ok<NASGU>
            pause(pause_duration);
            hold off;
        end
    end
end

