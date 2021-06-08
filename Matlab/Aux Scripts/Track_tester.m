%% Note: Execute this script after the track_creator script 
%       (or, if the tracks are created already, after loading the vars.mat)


%% Manually compare orientations and positions

load('C:\Users\Morais\Documents\SXR_Project\SXRSIMv2\Matlab\Tracks\Track_SEED5_SPEED1_UES4.mat');
t1 = l_aux.rx_track(1,1);

load('C:\Users\Morais\Documents\SXR_Project\SXRSIMv2\Matlab\Tracks\Track_SEED5_SPEED1_UES4v2.mat');
t2 = l_aux.rx_track(1,1);

p1 = t1.positions;
p2 = t2.positions;
o1 = t1.orientation;
o2 = t2.orientation;

%% Visually compare orientations and positions for different speeds
% load('vars_4ues.mat'); 

% SEED = 1;
% snapshot_interval = 8;
% display_orientation = 1;
% pause_interval = 17;
% pause_duration = 5;
% shade = 0;
% make_gif(4) = 1;

SEED = 1;
snapshot_interval = 100;
display_orientation = 1;
pause_interval = 3;
pause_duration = 2;
shade = 0;
make_gif(4) = 1;
plot_only_phy_users = 0;     % 0 means all phy + vir users are plotted.
plot_with_layout = 0;        % 1 if layout is needed in the gif.

for mvnt_val = 3:3  % check the speed here!
    filename = ['Tracks\Track', '_SEED', num2str(SEED), ...
                '_SPEED', num2str(mvnt_val), '_UE', num2str(n_rx), '.mat'];
    l_aux = load(filename).l_aux;
    
    for i = 1:1 % do for up only..
        if i == 1
            view_mode = "up"; zoom_factor = 1;
        else
            view_mode = '30 25'; zoom_factor = 1.2;
        end

        % l_aux = l % to plot the layout directly from the vars.mat
        plot_movement_all(l_aux, rx_pos_non_present, limits, n_users, ...
            update_rate, snapshot_interval, pause_interval, pause_duration, ...
            shade, view_mode, 2, save_plots(5), make_gif(4), ...
            display_orientation, zoom_factor, plot_only_phy_users, pop);
        
        N_users = n_users - size(other_participants,2);
        plot_track_users(l_aux, rx_pos_non_present, ...
            limits, N_users, update_rate, snapshot_interval,...
            pause_interval, pause_duration, shade, view_mode, ...
            make_gif(4), display_orientation, zoom_factor, ...
                                    plot_only_phy_users, plot_with_layout)

    end
end