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
pause_duration = 0;
shade = 0;
make_gif(4) = 1;
plot_only_phy_users = 0;     % 0 means all phy + vir users are plotted.
plot_with_layout = 0;        % 1 if layout is needed in the gif.

for mvnt_val = 4:4  % check the speed here!
    filename = ['Tracks\Track', '_SEED', num2str(SEED), ...
                '_SPEED', num2str(mvnt_val), '_UE', num2str(n_rx), '.mat'];
    filename = 'Tracks\Track_SEED1_SPEED4_UE4.mat';
    l_aux = load(filename).l_aux;
    
    for i = 1:1 % do for up only..
        if i == 1
            view_mode = "up"; zoom_factor = 1;
        else
            view_mode = '30 25'; zoom_factor = 1.2;
        end
        
        plot_track_users(l_aux, rx_pos_non_present, ...
                         limits, n_users, update_rate, snapshot_interval,...
                         pause_interval, pause_duration, shade, ...
                         view_mode, make_gif(4), display_orientation, ...
                         zoom_factor, plot_only_phy_users, plot_with_layout)
        
        
        plotCircle3D( [room_centre_xy cam_height-0.05], [0 0 1], ...
                      r_table, [0.5 0.25 0]);
    end
end