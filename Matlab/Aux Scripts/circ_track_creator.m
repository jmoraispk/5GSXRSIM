% This script creates and saves circular track of single user around a 
% table of radius 1.4 m. Other 3 phy users are seated around the table 
% facing the center of the table and radius of 1.6 m from the center of the 
% table. BS is located at a height of 3 m above the center of the table.


f = 26e09;              % Frequency
SEED = 1;
speed = 1;              % Enter walking speed of UE here (1 m/s or 2 m/s) 
len_track = 2 * 1.8 * pi;   % circular track 2 * pi* r circumference.
time_compression_ratio = 40;

%%%NUMBER OF SNAPSHOTS CALCULATION%%%%%%%

numerology = [ 2 ];
simulation_duration = 20;                        % in seconds
dist_tot = speed * simulation_duration;     % Total distance to be covered 
                                            % by the moving user.
no_circles = floor( dist_tot / len_track );
eff_displacement = dist_tot - ( no_circles * len_track );
base_update_rate = 0.001;              % duration of TTI for numerology 0.
update_rate = ( base_update_rate / ( 2 ^ max( numerology ) ) ) * ... 
                                            time_compression_ratio;
sim_duration_snapshot = round( simulation_duration / update_rate ); %#snapshots 

%%%%%%%%%%%%SIMUALTION PARAMETERS%%%%%%%%%%%%%%%%%

s = qd_simulation_parameters;
s.show_progress_bars = 1;
s.center_frequency = f;

%%%%%%%%%%NETWORK LAYOUT%%%%%%%%%%%%%%%%%%%%%%%%%

l_aux = qd_layout( s );
l_aux.no_rx = 4;                                          % 4 UEs
l_aux.no_tx = 1;                                          % 1 BS
l_aux.tx_position = [ 4; 4; 3 ];

%%%%%%%%%TRACK CONFIGURATION%%%%%%%%%%%%%%%%%%%%%

% Setting track, initial position and #TTIs covered based on UE-1 speed. 
name_aux = l_aux.rx_track( 1, 1 ).name;
l_aux.rx_track( 1, 1 ) = qd_track( 'circular', len_track, -pi / 2 );
l_aux.rx_track( 1, 1 ).name = name_aux;
l_aux.rx_track( 1, 1 ).initial_position = [ 4; 2.2; 1.6 ];

samples_per_second = 1 / update_rate;
TTIs_per_m = samples_per_second / speed;

% replicate positions before interpolation (for no_circles plus 2)
l_aux.rx_track( 1, 1 ).interpolate_positions(TTIs_per_m);
l_aux.rx_track( 1, 1 ).no_snapshots = sim_duration_snapshot + 1;
l_aux.update_rate = update_rate;

% Setting the name, position and orientation of the other 3 phy UEs, for
% now all set as 0s.
for i = 2 : l_aux.no_rx
    l_aux.rx_track(1, i).name = [ 'Rx000', num2str( i ) ];
    l_aux.rx_track(1, i).positions = ...
                            zeros(size( l_aux.rx_track(1, 1).positions ));
    l_aux.rx_track(1, i).orientation = ...
                           zeros(size( l_aux.rx_track(1, 1).orientation ));
end

l_aux.rx_track(1, 2).initial_position = [ 5.6; 4; 1.4 ];
l_aux.rx_track(1, 2).orientation(3, :) = ...
                          l_aux.rx_track(1, 2).orientation(3, :) + ( pi );
l_aux.rx_track(1, 3).initial_position = [ 4; 5.6; 1.4 ];
l_aux.rx_track(1, 3).orientation(3, :) = ...
                          l_aux.rx_track(1, 3).orientation(3, :)+(-pi / 2);
l_aux.rx_track(1, 4).initial_position = [ 2.4; 4; 1.4 ];
l_aux.rx_track(1, 4).orientation(3, :) = ...
                                    l_aux.rx_track(1, 4).orientation(3, :);

for k = 1 : l_aux.no_rx
    l_aux.rx_track(1, k).segment_index = 1;
    l_aux.rx_track(1, k).scenario = '3GPP_38.901_Indoor_LOS';
end

% Setting orientation of the UE on circular track to always point at the
% centre of table.
l_aux.rx_track(1, 1).orientation(3, :) = ...
                l_aux.rx_track(1, 1).orientation(3, :) + (-pi / 2);

%%%%%%% interpolation if track is repeated%%%%%%%
q1 = round( TTIs_per_m * len_track );   % total TTIs for 1 circular travel.
if l_aux.rx_track(1, 1).no_snapshots > q1  
    if no_circles > 1 %This if is to interpolate x repetitions of full circle track.
        for q = 1 : no_circles - 1
            l_aux.rx_track(1, 1).positions(:, ( (q * q1) + 1) : ( (q * q1) + q1)) = ...
                l_aux.rx_track(1, 1).positions(:, 1 : q1);
            l_aux.rx_track(1, 1).orientation(:, ((q * q1) + 1) : ( (q * q1) + q1 )) = ...
                l_aux.rx_track(1, 1).orientation(:, 1 : q1);
        end
    else
        % if no of circles is 1.x, then to interpolate the remaining 0.x
        % for postion and orientation of the moving user.
        tempx = size(l_aux.rx_track(1, 1).positions(:,((q1 + 1) : end)));
        l_aux.rx_track(1, 1).positions(:,( (q1 + 1) : end )) = ...
            l_aux.rx_track(1, 1).positions(:, 1 : tempx( 2 ) );
        l_aux.rx_track(1, 1).orientation(:,( (q1 + 1) : end )) = ...
            l_aux.rx_track(1, 1).orientation( :, 1 : tempx( 2 ) );
    end

end

% Visualizing & saving layout for track testing & further processing.
l_aux.visualize;
filename = ['Tracks\Circ_Track', ...
                    '_SEED', num2str(SEED), ...
                    '_SPEED', num2str(speed), ...
                    '_UE', num2str(l_aux.no_rx),...
                    get_time_str(), '_test_delete.mat'];

save(filename, 'l_aux');

