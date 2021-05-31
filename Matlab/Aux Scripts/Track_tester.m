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
load('vars_4ues.mat'); 

SEED = 1;
snapshot_interval = 8;
display_orientation = 1;
pause_interval = 17;
pause_duration = 5;
shade = 0;
make_gif(4) = 1;

for mvnt_val = 3:3  % check the speed here!
    filename = ['Tracks\Track', '_SEED', num2str(SEED), ...
                '_SPEED', num2str(mvnt_val), '_UE', num2str(n_rx), '.mat'];
    filename = 'Tracks\Circ_Track_SEED1_SPEED1_UE4_point_centre.mat';
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
            display_orientation, zoom_factor);

    end
end