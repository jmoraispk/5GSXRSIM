function [] = Meeting12(input_filename, flow_control, instance_info)

tic;

% All information about flow_control variable inside the format function
if isstring(flow_control) || ischar(flow_control)
    % Convert the string format to numbers format
    flow_control = format_flow_control(flow_control);
end

% All information about instance_info variable inside the format function
if ~isempty(instance_info)
    instance_info = format_instance_info(instance_info);
end



if flow_control <= 2
    
    % Attempt to read input parameters from file
    
    % If it doesn't work, use the default parameters
    
    if isfile(input_filename)
         % File exists.
         disp('Input file loaded successfully!');
         load(input_filename); %#ok<LOAD> %load all!
         
    else
    
    disp("COULDN'T FIND THE INPUT FILE NAME. USING DEFAULT VALUES!!!!!!!");
    % Default values should be separated from calculations on these values
    % Firstly, all inputs default values, then all computations
    
    %% PARAMETERS
    SEED = 12;
    
    debug_mode = [0; % Head Model working mode 
                  0; % Positions, Tracks & Movement
                  0; % Scenarios in Stats 
                  0; % Speaker list
                  0; % Print speaker list azimuth (right target direction?)
                  1];% Print steps of the way.
    
    if debug_mode(6)
        disp('Starting Parameter Setup.');
    end
    
    %%%%%%%%% Folders %%%%%%%%%
    
    % a global variable named config_folder will be declared further ahead
    % it's used to tell QuaDRiGa where are the scenario configurations
    curr_dir = pwd();
    conf_folder = [curr_dir, '\QuaDRiGa\quadriga_src\config'];
                 
    % Only used in case 5 in order to fetch the channels of each instance
    channel_parts_folder = ['\Channel_parts\'];
    
    builders_folder = ['\Builders\'];
    
    default_builder_name = 'builder';
    
    override_existing_folder = 1;
    output_folder_name = ['Sim_', get_curr_time_str(), '\'];
    % the output filename will be the name of the folder and this should be
    % done in python. Then, we simply use that folder to fetch the
    % variables, which will always be named 'vars.mat'.
    
    
    %%%%%%%%%%%%%%%%%%%%%%%% General Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%
    progress_bars = 1;
    turn_off_positions = 0;
    turn_off_orientations = 0;
    get_full_freq_response = 1;
    save_freq_response = 1;
    apply_blocking = 0;
    save_partial_channel_time_domain = 1;
    aggregate_full_channel_time_domain = [1,1];
    aggregate_coeffs_in_freq = [0,0];
    
    backup_pos_and_ori = 1;
    save_pos_and_ori_backup_separately = 0;
    variability_check = 0;
    visualize_clusters = 0;
    
    calc_stats = [0;  % Power calculations
                  0]; % Path calculations
    %%%%%%%%%%%%%%%%%%%% Visualization parameters %%%%%%%%%%%%%%%%%%%%%%%%%
    visualize_plots = [0; % Room Arrangement 
                       0; % Track users movement only
                       0; % Track users movement only (on layout)
                       0; % Track users movement + virtual users
                       0; % Track users movement + virtual users + BS (all)
                       0; % Track all (on layout)
                       1]; % Power plots per receiver (cameras too)
    save_plots =      [""; % Room Arrangement 
                       ""; % Track users movement only
                       ""; % Track users movement only (on layout)
                       ""; % Track users movement + virtual users
                       ""; % Track All
                       ""; % Track All (on Layout)
                       ""];% Power plots per receiver (cameras too)
    make_gif =        [0; % Track users movement only
                       0; % Track users movement only (on layout)
                       0; % Track users movement + virtual users
                       0; % Track users movement + virtual users + BS (all)
                       0];% Track all on layout 
    custom_limits = [0 4 0 4 0 4]; 
    enable_automatic_limits = 1; % if enabled, auto_limits are used.
    snapshot_interval = 500;
    pause_interval = 1;
    pause_duration = 1;
    shade = 1;
    display_orientation = 0;
    view_mode = "normal"; %'up', 'side'
    %%%%%%%%%%%%%%%%%%%%% More General Parameters %%%%%%%%%%%%%%%%%%%%%%%%%


    simulation_duration = 0.02; %[s]
    begin_slack = 1; %trim the first x-1 samples

    f_values = [3.5 26] * 1e9; %#ok<*NBRAK> % [Hz] 
    %For mmWave frequencies, there may appear an warning regarding 
    %sample density. Fulfilling the sampling theorem is only important if
    %we want to interpolate at the maximum speed after creating the trace,
    %which does not happen. Thus, the warning is passible of being ignored.
    
    base_update_rate = 1e-3; % update rate of the base numerology - num0.
    
    % If the final update rate is to be derived from the numerology:
    time_compression_ratio = 1;
    
    % IF the variable above is set to 1, there will be no compression in
    % time. But if the value in bigger than one, then each sample in 
    % time will corresponde to those many TTIs!
    
    
    % IMPORTANT! KEEP THE NUMEROLOGY, BANDWIDTHS AND PRBS WITH THESE
    % DIMENSIONS, CHANGING BANDWIDTHS TO 0 IF THAT NUMEROLOGY IS NOT USED.
    
    % Insert only the numerologies to be simulated!
    numerology = [2]; % 5G is between 0 and 3 (for data)
    
    
    % Bandwidths to be used in each frequency at each numerology
    bandwidth = [25; 
                 25;] .* 1e6;
    
    % Number of subcarriers in each frequency, at numerology
    % This can be derived from the subcarrier spacing and bandwidth,
    % but this way we keep all 5G complexity in Python, and allow this
    % simulator to be used (or more easily converted ) in other radio 
    % technologies.
    n_prb = [34;
             34;]; 
    
    % In Python the math is simply: bandwidth / PRB_bandwidth, with
    % PRB_bandwidth = subcarrier_spacing * 12;
    
    % In case we would like to drift from this and simulate less PRBs, 
    % include the variable below to notify the simulation
    % (The compression rate must be an integer and represents how many PRBs
    % are contained in each sample in frequency
    freq_compression_rate = 1;
    
    % shortcuts to implement eventually:
    
    %use_same_bandwidth_all_freq = 1;
    %use_same_bandwidth_all_num = 1;
    %use_same_n_carriers_all_freq = 1;
    %use_same_n_carriers_all_num = 0;
    
    
    
    n_room(1) = 1; %length of phy_user_pos
    n_room(2) = 2; %length of vir_user_pos
    n_users = sum(n_room);
    n_camera_streams = 0; %cameras per user                    

    n_tx = 1;                                       % Number of BSs

    tx_height = 3;   % [m]
    user_height = 1.2; % is 1.3 is more realistic?
    cam_height = 0.80;  % [m]

    use_standard_arrangement = 1;
    % quadrangular, rectangular or circular
    table_type = 'round';
    % For rectangular room:
    seat_arrangement = [1 1];
    total_seats = 16; % 2*sum(seat_arrangement); %used for round table

    %angle of 30 degrees from the user's perpendicular to each camera
    d_u = 0.6;
    d_s = 0.3;

    %phy_user_disposition = 'uniform'; %[] for random, or pick manually
    phy_user_disposition = [0];% 'uniform';%[1];
    vir_user_disposition = [4 12];

    %Note: picking uniform for phy and certain places for vir may not 
    %work as intended, due to priority clashing. vir will be random if 
    %phy is uniform

    %set custom arrangement
    phy_usr_pos = [ 4; 3 ; 1.2];    %#ok<*NASGU> %UE positions
    cam_pos = [];

    tx_pos = [ 3; 3; tx_height];

    vir_usr_pos = [2 4; 4 3; 1 1]; 
    
    % Based on the combination of number of TXs and this variable, they
    % are placed in different places. 
    % 0 - no special places, just put them at the 'tx_pos' position
    % 1 - 'centre first': places BS at the centre first, and if there's
    %                     more than one, they are placed in the corners
    %                     of the room. Requires 'room_size' variable!
    % 2 - 'corners first': places BS at the corners first. If there's more
    %                      than 4, puts at the centre.
    % Note: max number of TXs (BSs) to be placed automatically is 5!
    %       For n_tx > 5, the positions need to be set in tx_pos.
    tx_special_pos = 1; 
    
    % To be used with tx_special_pos to defined the prefered corners 
    % of the room
    select_corners = [1 2 3 4]; % [1 2 3 4] is the normal order.
                                % [3 4 1 2] would favour corner 3, then 4,..
    % Note: corner 1 is ... and the rest are counter-clock wise.
    

    % Room dimensions
    room_size = [6, 6];
    
    % The offset from the centre of mass of the head of a user, (x,y,z)in m
    rx_pos_offset = [0.15 0 0.05];
    % The degrees to rotate the linear antenna. By default, it's vertical.
    rx_rot_offset = 90;

    % load_tracks
    load_tracks_from_file = 0;
    
    % Radius of the table, for user placement purposes
    % (it can't be smaller than half of any room dimension
    r_table = 1;
    % User distance from the centre of the table.
    r_users_dist = 0.2;
    %%%%%%%%%%%%%%%%%%%%%%%% Movement parameters %%%%%%%%%%%%%%%%%%%%%%%%%%
    % Radius of sphere for random head position of the participants
    r_head_sphere = 0.2; 
    %such that 99.7%  of positions are inside this sphere
    sigma_head = r_head_sphere/3; 

    %Position
    speed_profile = 0;       % 0- Constant speed across the simulation
                             % 1- Vary mvnt profiles across the simulation
    
    % define the movement profiles of the receivers
    % How quick each receiver moves (Position & Orientation)
    rx_mvnt = [1, 4, 7];  
    %    Moving speed      Rotation Interval
    %        0                      -         (static) for cameras
    % 1      0.1                    1.3
    % 2      0.2                    1.1
    % 3      0.3                    0.9
    % 4      0.4                    0.7
    % 5      0.5                    0.5
    % 6      0.6                    0.3
    % 7      0.7                    0.1
    % table implemented in function get_speed_and_rot_int

    same_mvnt_for_all_usrs = 1; 
    const_mvnt_value = 3;
    

    %Rotation
    head_model_rotation = 6;
      % 0- No rotation, only position change
      % 1- Random, uniform between the limits
      % 2- Random, bet. limits, with intervals
      % 3- Cone to participants, prob. of wandering = 0
      % 4- Cone to participants, staring
      % 5- Look to who's talking model, staring used as speech dur.
      % 6- Look at the person that spoke last, to emulate answering.
    
    same_head_model_across_users = 1;
    
    % probability of maintaining the look is 1/staring_param
    staring_param = 10e9; 
    %for model 5&6:
    % We can create the speaker list stochastically. Set the avg_time
    % variable for that.
    speaking_avg_time = 2; 

    bank_lim = pi/9; %slack on nose axis, from shoulder to shoulder
    tilt_lim = pi/6; %slack on ears axis, saying yes and no
    fdang = pi/72; %focus drift angle - azimuth angle slack.

    custom_speaker_list_uses_seats = 0;
    use_custom_speaker_list = 0;
    only_vir_speak = 1;
    % if the list is to be constructed automatically: time for each speech
    speaking_time = 2.1; % [s]
    
    
    % The speaker list is [Time of beginning speach, speaker_idx]
    % Note: the speaker index is the index of the user in all_user_pos
    % all_user_pos = [phy_usr vir_usr]. 
    % E.g. If the 3rd virtual user speaks,
    % and there are 4 physical users, the index is 7.
    custom_speaker_list = [0 2; 2 3; 14 2];
        
    %%%%%%%%%%%%%%%%%%%%%%%% Antenna configurations %%%%%%%%%%%%%%%%%%%%%%%
    % omni, dipole, patch or array
    
    % A row per frequency, a column per UE/BS. Put a single column if the
    % same antenna is to be used across users, cameras or BSs
    
    user_ant = ["array"; "array"];
    cam_ant = ["array"; "array"];
    bs_ant = ["array"; "array"];
    
    % IMPORTANT NOTE: the 3 options above are overriden in accordance with
    %                 the values on the following variables:
    % e.g. if same_antenna_across_ues = 1, cam_ant will be = to user_ant.
    % e.g. same_antena_across_users is 1, then only one value is expected
    %      for the user_ant variable, and will be repeated across users
    same_antena_across_users = 1;
    same_antena_across_cams = 1;
    same_antenna_across_ues = 0;
    same_antena_across_txs = 1;
    same_antenna_across_frequencies = 1;
    same_antenna_cam_user = 0;
    
    % 0 means only one polarisation per dual-polarised element
    % 1 results in taking the polarisations separately
    diff_orthogonal_polarisation = 1;
    
    % Inputs Required
    % The antenna configurations are only used if the 'array' antenna
    % type is selected (# elements V, # elements H)
    % BS
    bs_ant_config = [4 4;
                     4 4];
    bs_ant_element_spacing = 0.5;
    % USER
    user_ant_config = [2 2;
                       2 2];
    user_ant_element_spacing = 0.5;
    % CAM
    cam_ant_config = [1 1;
                      2 2];
    cam_ant_element_spacing = 0.5;
    
    
    % Hybrid Beamforming paramters
    % The above configurations are always the total number of AEs
    
    % whether to enable it
    bs_ant_hybrid_bf = 0;
    user_ant_hybrid_bf = 0;
    cam_ant_hybrid_bf = 0;
    % whether to aggregate responses or not 
    bs_one_response_per_subarray = 1;
    user_one_response_per_subarray = 1;
    cam_one_response_per_subarray = 1;
    % subarray structure (note: only vertical subarrays are possible
    bs_ant_subarray = [1 1;
                       4 1];
    user_ant_subarray = [1 1;
                         3 1];
    cam_ant_subarray = [1 1;
                        2 1];
    
    
    %%%%%%%%%%%%%%%%%%% SEGMENT & SCENARIO SETTINGS %%%%%%%%%%%%%%%%%%%%%%%
    manual_LOS = 0; %This is done to simulate blockages
    %probabilistic_scen = '3GPP_38.901_Indoor_Open_Office';
    LoS = '3GPP_38.901_Indoor_LOS';
    NLoS = '3GPP_38.901_Indoor_NLOS';
    %LoS_only = 'LOSonly';
    
    default_scen = LoS;
    %default_scen = 'LOSonly';
    %see justification on the report, scenario section
    
    %what overlap percentage should be considered?
    overlap_percentage = 0.15;
    %if manual_LOS = 0, overlap_percentage is not used
    
    % Percent of the start of scenarios, from the second onwards, relative
    % to the full simulation duration (e.g. 0.05-> start at 0.05 * sim_dur) 
    segment_start = [];
    % needs to be one more than the segment start because the first
    % scenario is included as well
    scenarios = {LoS}; 
    
    use_same_scenarios_all_users = 1;
    
    builders_to_generate = 'all';
    end
    
    

    %% %% %% %% %% %% Parameter calculation/derivation %% %% %% %% %% %%
    % these parameters should be computed from the input parameters
    
    % I'm sorry, this needs to be accessed in a file deep in QuaDRiGa.
    % It was the only way of not changing the arguments of 10 functions.
    if exist('conf_folder', 'var') %for variables
        global config_folder; %#ok<TLEV>
        config_folder = conf_folder;
    end
    
    
    % SET SEED! This is done for the random number generations that happen
    % throughout the track generation process. Random head positions, etc..
    rng(SEED, 'twister');
    
    %%%%%%%%%%% Create folder where everything will be saved %%%%%%%%%%%%
    % Folder that will contain everything on this simulation.
    
    % Everything respecting this simulation will be inside this folder
    % The name is the current date and time
    
    mother_folder = output_folder_name;
    if exist(mother_folder, 'dir')
        if override_existing_folder
            % delete the folder with the same name
            rmdir(mother_folder, 's');
        else
            error(['The folder already exists!!! ', ...
              'Re-doing the MatlabInput.mat generation, ', ...
              'or deleting it should solve it.']);
        end
    end
    
    if debug_mode(6)
        disp(['Creating Simulation Folder at:', mother_folder]);
    end
    
    mkdir(mother_folder);
    
    
    % If the input was provided, copy it to the simulation folder
    if ~isempty(input_filename)
        copyfile(input_filename, mother_folder)
    end
    
    
    % These are only actually used in parallel, but they need to be set to 
    % allow calling of common functions
    builders_folder = [mother_folder, builders_folder];
    channel_folder = [mother_folder, channel_parts_folder];
    channel_blocked_folder = [channel_folder(1:end-1), ...
                                      '_blocked\'];
    if flow_control == 2    
        % And create auxiliar directories to store all sorts of stuff
        mkdir(builders_folder);
        mkdir(channel_folder);
        
        % Make a separate folder for the channel parts from blocking.
        if apply_blocking
            mkdir(channel_blocked_folder);
        end
    end
    
    % Numerology, PRB and bandwidth verifications
    
    n_freq = length(f_values);
    if size(bandwidth, 1) ~= n_freq || size(n_prb, 1) ~= n_freq
        error(['Carrier Bandwidth and number of prbs per carrier need', ...
              ' to have as many rows as there are frequencies.']);
    end
    
    if size(bandwidth, 2) ~= length(numerology) || ...
           size(n_prb, 2) ~= length(numerology)
        error(['Carrier Bandwidth and number of prbs per carrier need', ...
              ' to have as many columns as there are numerologies.']);
    end
    
    if ~isequal(non_zero_numerologies(numerology, bandwidth), numerology) || ...
       ~isequal(non_zero_numerologies(numerology, n_prb), numerology)
        error(['All numerologies must be used and have valid prbs ', ... 
               'and bandwithds']);
    end
    
    
    %%%%%%%%%%%% General %%%%%%%%%%%%%%%
    
    % Compute update rate based on numerology
    update_rate = base_update_rate / (2^max(numerology));
    % Adjust update rate based on time compression
    % i.e. there will be time_compression_ratio times less time instants! 
    update_rate = update_rate * time_compression_ratio;
    
    % Very importantly: in case we are doing any time compression, we need
    % to include an extra sample to enable interpolation
    if time_compression_ratio ~= 1
        extra_sample = 1;
    else
        extra_sample = 0;
    end
    
    n_rx = n_room(1) * (1 + n_camera_streams);    % Number of UEs
    
    
    %%%%%%%%%%%%% Position %%%%%%%%%%%%
    r_users = r_table + r_users_dist;
    
    cam_parameters = [d_u d_s 0];
    % The third argument is for having other cameras in the room
    % 0 means 'no', because they aren't implemented anyway.
    heights = [user_height, cam_height, tx_height];
    room_centre_xy = room_size / 2;
    
    if use_standard_arrangement
        % Make verifications to (phy) and (vir) _user_disposition
        if (isstring(phy_user_disposition) && ...
            ~strcmp(phy_user_disposition, 'uniform') ) || ...
            (~isstring(phy_user_disposition) && ...
             ~all(phy_user_disposition >= 0 & ...
             phy_user_disposition < total_seats))
                error(['Possible values for phy and virtual user ', ...
                       'dispositions are arrays of integers between ', ...
                       '0 and total_seats-1']);
        end      
        if (isstring(vir_user_disposition) && ...
            ~strcmp(vir_user_disposition, 'uniform') ) || ...
            (~isstring(vir_user_disposition) && ...
             ~all(vir_user_disposition >= 0 & ...
             vir_user_disposition < total_seats))
                error(['Possible values for phy and virtual user ', ...
                       'dispositions are arrays of integers between ', ...
                       '0 and total_seats-1']);
        end      
            
        
        % Do room verifications
        
        %Physically present, virtually, cams
        [phy_usr_pos, vir_usr_pos, cam_pos] = ...
            room_arrangement(table_type, room_size, seat_arrangement, ...
                             total_seats, n_room(1), n_room(2), ...
                             n_camera_streams, cam_parameters, 1.0, ...
                             phy_user_disposition, vir_user_disposition,...
                             heights(1:2), r_table, r_users);

        
        tx_pos = place_BSs(n_tx, tx_pos, tx_special_pos, room_size, ...
                           tx_height, select_corners);
        
    end
    
    
    % present UEs
    rx_pos = [phy_usr_pos cam_pos];

    % Virtual/non-present participants
    rx_pos_non_present = [vir_usr_pos];

    % All users
    all_users_pos = [phy_usr_pos vir_usr_pos];
    
    if visualize_plots(1)
        position_check_plot(phy_usr_pos, vir_usr_pos, tx_pos, ...
                            n_camera_streams, cam_pos);
    end

    if ~check_size(rx_pos, [3 n_rx]) || ~check_size(tx_pos, [3 n_tx])
        error("Amount of positions doesn't match amount of RX or TX");
    end
    
    % Automatic limits computation:
    if enable_automatic_limits
        % Every position
        everything_pos = [rx_pos rx_pos_non_present tx_pos];
        
        % Margin around the edges
        mx = 0.5;
        my = 0.5;
        mz = 1;
        limits = ...
         [min(everything_pos(1,:)) - mx, max(everything_pos(1,:)) + mx,...
          min(everything_pos(2,:)) - my, max(everything_pos(2,:)) + my,...
          min(everything_pos(3,:)) - mz, max(everything_pos(3,:)) + mz];
    else
        limits = custom_limits %#ok<*UNRCH>
    end
    
    
    %%%%%%%%%%%%% Movement %%%%%%%%%%%%
    r_cam = 1e-10; %for camera stillness (ignore)
    
    speed_profile = 0;
    if speed_profile == 1
        disp(['Varying movement profiles are not implemented.', ...
              '(No energy/mood/pattern change in terms of head speed', ...
              ' profiles. Someone fast is fast during the whole simulation', ...
              ', someone slow is always slow as well']);
        return
    end
    
    % n_mvnt_samples is an auxiliar to compute the number of macro 
    % positions to be generated and interpolated. Most of them will be 
    % discarded we trim the ones that made it into the simulation duration.
    n_mvnt_samples_aux = get_empirical_n_mvnt_samples(simulation_duration, ...
                                                      r_head_sphere);
    
    %%%%%%%%%%%%% Head Model %%%%%%%%%%%%
    if same_mvnt_for_all_usrs
        rx_mvnt = repmat(const_mvnt_value, [1 n_room(1)]);
    end
    rx_mvnt = [rx_mvnt zeros([1, size(cam_pos, 2)])];
    
    if same_head_model_across_users
        head_model_rotation = repmat(head_model_rotation, [n_room(1) 1]);
    end
    
    if size(head_model_rotation,1) ~= n_room(1)
        error('head_model_rotation has wrong dimensions!');
    end
    
    %only for head_model_rotation = 5;
    if use_custom_speaker_list
        speaker_list_aux = custom_speaker_list;
        
        if custom_speaker_list_uses_seats
            speaker_list_aux(:,2) = ...
              map_seats_to_user_idxs(speaker_list_aux(:,2), ...
                                     phy_user_disposition, ...
                                     vir_user_disposition)
                                 
            speaker_list = speaker_list_aux;
        else
            speaker_list = speaker_list_aux;
            % when user indices are chosen, we need to sum one because
            % python is 0-indexed and matlab is not. 
            speaker_list(:,2) = speaker_list(:,2) + 1;
        end
        
    else
        % non-custom speaker list assumed 3 things: 
        %   - as many physical as virtual users, with no two physical users
        %     seated adjacent to one another, and the same for virtual
        %   - the same speaking time for each
        %   - a given pattern for the talk: 
        %       a) user 0 (phy) starts
        %       b) phy and vir always switching: phy->vir->phy->vir->etc.
        %       c) the next user is the 2nd going around the circle to 
        %          counterclockwise, which consists in increments of 3
        %          in the index, when users are seated P-V-P-..
        %   - if only_vir_speak is True, then ... only the virtual users
        %     speak. One after the other, the same as above.
        
        % NOTE: user_order should have user indices, not seats.
        %       E.g. idxs 0-room(1) are for physical users, and from that 
        %            idx on is for virtual users!
        % In fact, all_users_pos = [rx_pos rx_pos_nonpresent]
        
        if only_vir_speak
            % Virtual users only!
            user_order = n_room(1)+1:n_users;
            speaking_instants = (0:n_room(2)-1) * speaking_time;
            speaker_list = [speaking_instants', user_order'];
        else
            % Assuming the same number of phy and vir!
            seat_order = mod((0:n_users-1) * 3, n_users);
            
            mapping_to_users = map_seats_to_user_idxs(seat_order, ...
                                                      phy_user_disposition, ...
                                                      vir_user_disposition)
            
            speaking_instants = (0:n_users-1) * speaking_time;
            speaker_list = [speaking_instants', mapping_to_users'];    
        end
        
    end
    
    % Final speaker list touches:
    % A) add extra speaker at the end, to guarantee enough positions are
    % generated
    speaker_list = [speaker_list; ...
                    (simulation_duration+1) 0];

    if debug_mode(4)
        disp(speaker_list)
    end
    
    if  max(head_model_rotation) >= 5  && ...
        (speaking_avg_time * 1.5) > simulation_duration && ...
        use_custom_speaker_list == 0 
        % only give warning if auto speaker generation is enabled
        warning("Risk of only one speaker in this simulation...");
    end
    
    %%%%%%%%%%%%% Antennas %%%%%%%%%%%%
    
    if same_antenna_across_ues
        cam_ant = user_ant;
    end
    
    if same_antena_across_users
        user_ant = repmat(user_ant, [1 n_room(1)]);
    end
    
    if same_antena_across_cams
        if n_camera_streams > 0
            cam_ant = repmat(cam_ant, [1 n_room(1)*n_camera_streams]);
        end
    end
    
    if same_antena_across_txs
        bs_ant = repmat(bs_ant, [1 n_tx]);
    end
    
    
    rx_ant = [user_ant cam_ant];
    tx_ant = [bs_ant];
    %rx and tx ant need to be (n_rx or n_tx) x n_freq
    
    if same_antenna_across_frequencies
        if n_freq > 1 %have the same antennas for every frequency
            rx_ant = repmat(rx_ant, [n_freq 1]); 
            tx_ant = repmat(tx_ant, [n_freq 1]);
        end
    end
    
    
    if same_antenna_cam_user %usr and cameras have the same antenna conf
        cam_ant_config = user_ant_config;
        cam_ant_element_spacing = user_ant_element_spacing;
    end
    
    % Note, rx_ant and tx_ant have: 
    %  Number of columns = n_freq
    %  Number of rows = n_tx or n_rx (tx_ant or rx_ant, resp.)
    
    
    % If that wasn't the case, this would be how to get the size of each 
    % antenna
    rx_ant_numel = zeros(size(n_freq, n_rx));
    tx_ant_numel = zeros(size(n_freq, n_tx));
    for f_idx = 1:n_freq
        for rx_idx = 1:n_rx
            if isequal(rx_ant(f_idx, rx_idx), {'array'})
                if rx_idx <= n_room(1)
                    rx_ant_numel(f_idx, rx_idx) = ...
                                            prod(user_ant_config(f_idx,:));
                else
                    rx_ant_numel(f_idx, rx_idx) = ...
                                            prod(cam_ant_config(f_idx, :));
                end
                if diff_orthogonal_polarisation
                    rx_ant_numel(f_idx, rx_idx) = ...
                                           rx_ant_numel(f_idx, rx_idx) * 2;
                end
            else
                rx_ant_numel(f_idx, rx_idx) = 1;
            end
        end
    end
    
    for f_idx = 1:n_freq
        for tx_idx = 1:n_tx
            if isequal(tx_ant(f_idx, tx_idx), {'array'})
                tx_ant_numel(f_idx, tx_idx) = ...
                                             prod(bs_ant_config(f_idx, :));
                if diff_orthogonal_polarisation
                    tx_ant_numel(f_idx, tx_idx) = ...
                                           tx_ant_numel(f_idx, tx_idx) * 2;
                end
            else
                tx_ant_numel(f_idx, tx_idx) = 1;
            end
        end
    end

    
    %%%%%%%%%%%%% Segment/Scenario setup %%%%%%%%%%%%
    if manual_LOS == 1
        sim_duration_snapshot = simulation_duration/update_rate;

        % assign segments based on the snapshots;

        %for each user, attribute scenarios to the segments
        
        for n = 1:n_rx
            if use_same_scenarios_all_users
            
                segment_info(n).segment_index = ...
                    [1, segment_start * sim_duration_snapshot]; %#ok<AGROW>
                segment_info(n).scenario = scenarios; %#ok<AGROW>
            else
                segment_info(n).segment_index = [1, ...
                               segment_start(n,:) * sim_duration_snapshot];
                segment_info(n).scenario = scenarios(n, :);
            end
        end
        %if length(segment_index) ~= scenario, then we have a problem
        for k = 1:n_rx
            if length(segment_info(k).segment_index) ~= ...
               length(segment_info(k).scenario)
                error(['Number of scenarios and indexes ', ...
                       'doesn''t add up for user ', num2str(k)]);
            end
        end

        max_num_segments = 1;
        for k = 1:n_rx
            if max_num_segments < length(segment_info(k).scenario)
                max_num_segments = length(segment_info(k).scenario);
            end
        end
    else
        %code for just one segment, the default.
        max_num_segments = 1;
        for n = 1:n_rx
            segment_info(n).segment_index = [1]; %#ok<AGROW>
            segment_info(n).scenario = {default_scen}; %#ok<AGROW>
        end
    end
    
    if flow_control == 3 && ...
            (get_full_freq_response == 0 || save_freq_response == 0)
        disp('Bear in Mind that NOTHING will be saved...');
    end
    
    
    % Calculate amount of snapshots in the simulation (before adjusting
    % the TTI delay (done after the scenarios)
    sim_duration_snapshot = round(simulation_duration/update_rate);
    
    % The parallelisation multiplier is equivalent to the number of
    % instances per time division, basically, the gain we can have by doing
    % parallelisation, counting that we parallelise on a time division
    % basis, which is not required also
    if flow_control > 0
        if ischar(instance_info(1))
            disp(instance_info(1));
        end
        
        parallelisation_level = instance_info(1);
        if parallelisation_level == 0
            n_instances_per_time_division = 1;
        elseif parallelisation_level == 1
            n_instances_per_time_division = n_freq;
        elseif parallelisation_level == 2
            n_instances_per_time_division = n_freq * n_tx;
        elseif parallelisation_level == 3
            n_instances_per_time_division = n_freq * n_tx * n_rx;
        else
            error('Only parallelisation levels from 0 to 3 are available');
        end

        %Here, instance_info(2) should be the number of partitions we'll
        % do to each builder.
        n_time_divisions = instance_info(2);

        if rem(n_time_divisions,1) % remaining
            disp('There''s adfad');
            error('The number of time divisions must be an integer!');
        end
        
        % IMPORTANT CHECK
        % since the frequency response for different numerologies will have
        % a smaller number of ttis, we must be sure of 2 things:
        %   a) that the total amount of ttis is divisible by 2,4 or 8, 
        %      (depending on the numerologies considered), in order to 
        %      make this reduction; 
        %   b) When we want to do time divisions, each division must also
        %      be divisible by this number, or else there will be a
        %      numerology that gets cut off because there won't be enough
        %      samples to average from.
        
        maximum_tti_compression = 2^(max(numerology)-min(numerology) - 1);
        
        % Check a)
        if rem(sim_duration_snapshot, maximum_tti_compression)
            error(['The number of time divisions will cause a ', ...
                   'problem when segmenting in different numerology']);
        end
        
        % Check b)
        if rem(sim_duration_snapshot / n_time_divisions, ...
                                                   maximum_tti_compression)
            first_poss_time_divs = find_divisors(sim_duration_snapshot);
            possible_time_divisions = first_poss_time_divs(...
                rem(sim_duration_snapshot./first_poss_time_divs, ...
                                              maximum_tti_compression)==0);
            error(['The simulation duration is not divided well ', ...
                   'by the current number of time divisions. ', ...
                   'Instead, pick one of the following: ', ...
                   num2str(possible_time_divisions)]);
        end
        
        % In the setup phase, instance_info is the number of time divisions
        
        % Trim idx has the extremes of each interval 
        trim_idx = linspace(0, sim_duration_snapshot, ...
                                     n_time_divisions + 1);
        % the +1 is accounting for the 0 that will be replaced by a 1
        trim_idx = [1 trim_idx(2:end)];
    end
    
    % calculate amount of snapshots in the simulation (final calculation)
    sim_duration_snapshot = round(simulation_duration/update_rate);
        
    %%Creating simulation parameters
    sim_param = qd_simulation_parameters;
    sim_param.show_progress_bars = progress_bars;
    sim_param.center_frequency = f_values;         % Set center frequency

    l = qd_layout(sim_param);
    
    % set default scenario. If the scenarios were set manually, they will
    % be attributed further below
    if ~manual_LOS
        l.set_scenario(default_scen); 
    end
    
    
    %%%%%%%%%%%%%%%%%%%% Antenna Arrays Definition %%%%%%%%%%%%%%%%%

    l.no_rx = n_rx;
    l.no_tx = n_tx;
    
    l.rx_position = rx_pos;
    l.tx_position = tx_pos;
    
    antenna_setup(l, f_values, rx_ant, tx_ant, ...
                  diff_orthogonal_polarisation, n_room, ...
                  user_ant_config, user_ant_element_spacing, ...
                  cam_ant_config, cam_ant_element_spacing, ...
                  bs_ant_config, bs_ant_element_spacing, ...
                  room_centre_xy, tx_height, rx_pos_offset, rx_rot_offset, ...
                  bs_ant_hybrid_bf, user_ant_hybrid_bf, cam_ant_hybrid_bf, ...
                  bs_ant_subarray, user_ant_subarray, cam_ant_subarray, ...
                  bs_one_response_per_subarray, ...
                  user_one_response_per_subarray, ...
                  cam_one_response_per_subarray);

    
    if debug_mode(6)
        disp(['Variable Setup is concluded. ', ...
              'Proceeding to create/load tracks (positions and orientations).']);
    end
    %% Positions, Tracks & Movement

    if load_tracks_from_file
        disp('Loading tracks...')
        % The track filename comes from Python
        load_tracks(l, tracks_filename);
    else
        disp('Creating tracks...');
        % Create tracks based on the many parameters
        create_tracks(l, head_model_rotation, bank_lim, tilt_lim, n_users, ...
                      sigma_head, all_users_pos, rx_pos, r_cam, n_rx, ...
                      n_mvnt_samples_aux, rx_mvnt, sim_duration_snapshot, ...
                      update_rate, begin_slack, fdang, speaker_list, ...
                      staring_param, speaking_avg_time, debug_mode, ...
                      simulation_duration, extra_sample, segment_info, ...
                      turn_off_orientations, turn_off_positions)
    end
    
    % Copy Positions and Orientations (for analysis in Python)
    if backup_pos_and_ori
        [pos_backup, ori_backup, initial_pos_backup] = ...
            get_pos_and_ori_from_layout(l); %#ok<ASGLU>
        
        if save_pos_and_ori_backup_separately
            % Save to folder
            save([folder, '/pos_ori_backup.mat'], ...
                 'pos_backup', 'ori_backup', 'initial_pos_backup');
        end
        % it is always saved in the vars.mat anyway.
    else
        [pos_backup, ori_backup, initial_pos_backup] = deal([0]);
    end
    
    if debug_mode(2)
        disp("After final trim;");
        disp(['Usable interval is from 1 to ', ...
                                    num2str(calc_max_snap(l.rx_track))]); 
        s1 = size(l.rx_track(1,1).positions,2);
        s2 = size(l.rx_track(1,2).positions,2);
        s3 = size(l.rx_track(1,3).positions,2);
        lens = l.rx_track(1,1:3).get_length;
        disp(['Sizes are: ', num2str(s1), ' ', num2str(s2), ...
                                          ' ', num2str(s3)]);
        disp(['Lens are: ', num2str(lens)]);
    end
    
    
    % Close your eyes for the following part.
    
    %Visualize only movement 
    if visualize_plots(2)
        plot_movement_all(l, rx_pos_non_present, limits, n_users, ...
            update_rate, snapshot_interval, pause_interval, pause_duration, ...
            shade, view_mode, 0, save_plots(2), make_gif(1), ...
            display_orientation, 1);
    end   
    %Visualize only movement with layout reference
    if visualize_plots(3)
        plot_movement_all(l, rx_pos_non_present, limits, n_users, ...
            update_rate, snapshot_interval, pause_interval, pause_duration, ...
            shade, view_mode, 1, save_plots(3), make_gif(2), ...
            display_orientation, 1);
    end
    %Visualize all users in it
    if visualize_plots(4)
        plot_movement_all(l, rx_pos_non_present, limits, n_users, ...
            update_rate, snapshot_interval, pause_interval, pause_duration, ...
            shade, view_mode, 2, save_plots(4), make_gif(3), ...
            display_orientation, 1);
    end
    %Visualize all users and BS.
    if visualize_plots(5) % THIS ONE IS A GOOD CHOICE!
        plot_movement_all(l, rx_pos_non_present, limits, n_users, ...
            update_rate, snapshot_interval, pause_interval, pause_duration, ...
            shade, view_mode, 3, save_plots(5), make_gif(4), ...
            display_orientation, 1); % 1 means no zoom
    end
    %Visualize complete representation on layout
    if visualize_plots(6)
        plot_movement_all(l, rx_pos_non_present, limits, n_users, ...
            update_rate, snapshot_interval, pause_interval, pause_duration, ...
            shade, view_mode, 4, save_plots(6), make_gif(5), ...
            display_orientation, 1);
        
        plotCircle3D( [room_centre_xy cam_height-0.05], [0 0 1], ...
                      r_table, [0.5 0.25 0]);
    end
    
    if debug_mode(6)
        disp(['Setup completed. Creating main builder.']);
    end
    
    original_builder = l.init_builder;
    
    % Set the random seed in order to have repeatable channel coeffs.
    rng(SEED, 'twister');
    
    % Force initialise the SoS-based generators.
    original_builder.init_sos(1);
    
    % This SoS object will be saved in case we need to parallelise
    
    if debug_mode(6)
        disp(['Main builder created.']);
    end
    
    if visualize_clusters
        %This is where cluster visualization can happen:
        builder_split(1).visualize_clusters(1, 2, 1);
        %scenario 1, for user 1, cluster 2, and create a new image;
    end
    
    % Now, the general layout is completely generated.
    
    
    %%%%%%%%%%%%%%%%%% IMPORTANT DIMENSIONS DEFINITIONS %%%%%%%%%%%%%%%
    % These will be used to separate and re-assemble the channel matrix
    
    
    % trick to get the paths, there's an angle of arrival for each path
    if debug_mode(6)
        disp('Generating parameters for main builder.');
    end
    if flow_control == 0    
        original_builder.gen_parameters;
    else
        original_builder(1).gen_parameters;
    end
    n_paths = size(original_builder(1).AoA, 2); 
    
    base_channel_dimensions = [n_rx, n_tx, n_freq];
    
    if flow_control == 2
        n_builders_per_time_div = prod(base_channel_dimensions);
        n_total_builders = n_builders_per_time_div * n_time_divisions;
    end
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

    if debug_mode(6)
        disp(['Saving Generation Variables.']);
    end
    % These are the things that should be the same in order to 
    % have the same outcome:
%         sos_obj.sos = original_builder.sos;
%         sos_obj.gr_sos = original_builder.gr_sos;
%         sos_obj.path_sos = original_builder.path_sos;
%         sos_obj.clst_dl_sos = original_builder.clst_dl_sos;
%         sos_obj.xpr_sos = original_builder.xpr_sos;
%         sos_obj.pin_sos = original_builder.pin_sos;


    % Save all variables appart from the function arguments
    save([mother_folder, '\vars.mat'], ...
                        '-regexp', ['^(?!(input_filename|', ...
                                         'flow_control|', ...
                                         'instance_info)$).']); 
                                         % removed ,'-v7.3'. 
    if debug_mode(6)
        disp(['Saved vars.mat in: ', mother_folder, '\vars.mat']);
        disp('Saving some variables to .txt');
    end
                                         
                                         
    % Fix a couple of vars before saving:
    if flow_control == 0
        n_instances_per_time_division = 0;
        n_builders_per_time_div = 0;
        n_time_divisions = 0;
        n_total_builders = 0;
        parallelisation_level = 0;
    end
    
    save_vars_to_txt([mother_folder, 'vars_summary.txt'], ...
                     {'SEED', 'aggregate_coeffs_in_freq', ...
                      'aggregate_full_channel_time_domain', 'bandwidth', ...
                      'bank_lim', 'base_channel_dimensions', 'bs_ant_config', ...
                      'cam_ant', 'cam_ant_config', 'const_mvnt_value', ...
                      'custom_speaker_list', 'custom_speaker_list_uses_seats', ...
                      'diff_orthogonal_polarisation', 'f_values', 'fdang', ...
                      'flow_control', 'get_full_freq_response', ...
                      'heights', ...
                      'instance_info', 'n_builders_per_time_div', ...
                      'n_camera_streams', 'n_instances_per_time_division', ...
                      'n_prb', 'n_room', 'n_rx', 'n_time_divisions', ...
                      'n_total_builders', 'n_tx', 'numerology', ...
                      'parallelisation_level', 'phy_user_disposition', ...
                      'r_table', 'same_antenna_cam_user', ...
                      'save_partial_channel_time_domain', 'select_corners', ...
                      'simulation_duration', 'tilt_lim', 'update_rate', ...
                      'use_custom_speaker_list', 'user_ant', ...
                      'user_ant_config', 'vir_user_disposition', ...
                      'time_compression_ratio', 'freq_compression_ratio',...
                      'builders_to_generate', 'load_tracks_from_file'}, ...
                     {SEED, aggregate_coeffs_in_freq, ...
                      aggregate_full_channel_time_domain,  bandwidth, ...
                      bank_lim, base_channel_dimensions, bs_ant_config, ...
                      cam_ant, cam_ant_config, const_mvnt_value, ...
                      custom_speaker_list, custom_speaker_list_uses_seats, ... 
                      diff_orthogonal_polarisation, f_values, fdang, ...
                      flow_control, get_full_freq_response, ...
                      heights, instance_info, ...
                      n_builders_per_time_div, n_camera_streams, ...
                      n_instances_per_time_division, n_prb, n_room, ...
                      n_rx, n_time_divisions, n_total_builders, n_tx, ...
                      numerology, parallelisation_level, ...
                      phy_user_disposition, r_table, same_antenna_cam_user, ...
                      save_partial_channel_time_domain, select_corners, ...
                      simulation_duration, tilt_lim, ...
                      update_rate, use_custom_speaker_list, user_ant, ...
                      user_ant_config, vir_user_disposition, ...
                      time_compression_ratio, freq_compression_ratio,...
                      builders_to_generate, load_tracks_from_file});

    if debug_mode(6)
        disp('Variables saved to .txt!');
    end 

    
    if flow_control == 2
        if debug_mode(6)
            disp(['Splitting main builder into one builder ', ...
                  'for each parallel instance.']);
        end
        % In case the simulation is meant to run in parallel:
        
        % First, trim the general layout into the layout of each part of
        % the channel, which corresponds
        
        for i = 1:n_time_divisions
            if ~strcmp(builders_to_generate, 'all') && ...
               ~ismember(i, builders_to_generate)
                continue
            end    
            
            l_t = l.copy;
            
            % Size of each instance (most of them should be the same, but
            % for certain simulation intervals and number of time divisions
            % it may happen they are not. Plus, the first is longer, this
            % helps with the organisation in the other side (Python)
            
            % Trim the tracks of every receiver
            trim_idx = int32(trim_idx);
            
            if i == 1
                first_idx = trim_idx(i);
            else
                first_idx = trim_idx(i) + 1;
            end
            
            last_idx = trim_idx(i+1) + extra_sample;
            
            for k = 1:l.no_rx
                l_t.rx_track(1,k).positions = ...
                 l.rx_track(1,k).positions(:, first_idx:last_idx);
                l_t.rx_track(1,k).orientation = ...
                 l.rx_track(1,k).orientation(:, first_idx:last_idx);
                % The initial_position stays the same, so that the tracks
                % have the same reference/origin
            end
            
            builder = l_t.init_builder;
            % Set the random seed to have repeatable and consistent coeffs.
            rng(SEED, 'twister');
            builder.init_sos(1);
            builder.gen_parameters;
%             builder.sos = sos_obj.sos;
%             builder.gr_sos = sos_obj.gr_sos;
%             builder.path_sos = sos_obj.path_sos;
%             builder.clst_dl_sos = sos_obj.clst_dl_sos;
%             builder.xpr_sos = sos_obj.xpr_sos;
%             builder.pin_sos = sos_obj.pin_sos;

            % Split builders and save them
            bs = split_multi_freq(builder);
            bs = bs.split_rx;
            
            save([builders_folder, default_builder_name, ...
                '_', num2str(i)], 'bs'); % we don't even include l_t!
            
            if debug_mode(6)
                disp(['Saved set of builder ', num2str(i), '/', ...
                      num2str(n_time_divisions)]);
            end
            
        end
    end
    
    
end
    %% Channel Calculation 
    %  (Calculate Channel coefficients in Time Domain, (save them), convert
    %   to frequency domain right away)
if flow_control == 0 || flow_control == 3 
    if flow_control == 3
        disp(['Starting Channel Calculation: Loading Variables']);
        % Load new variables (not caring for any overwites)
        load([input_filename, 'vars.mat']); %#ok<LOAD> 
    end
    
    if flow_control == 3
        % which time division? (there's a set of builders for each)
        
        
        [time_division_idx, relative_builder_idxs] = ...
            get_builder_idxs_from_instance(...
                instance_info(2), n_instances_per_time_division, ...
                n_builders_per_time_div);
        
        if debug_mode(6)
            disp(['Loading Builder from time division: ', ...
                                             num2str(time_division_idx)]);
        end
        
        builder_file = [builders_folder, default_builder_name, '_', ...
                        num2str(time_division_idx), '.mat'];
        
        % If the builder can't be found, simply print it and exit.
        % Python will handle this.
        if ~isfile(builder_file)
            disp(['There is no file with the name: ', builder_file]);
            disp('Builder not found! Terminating execution!!')
            return
        else
            % This Load will not colide with variables on the path
            load(builder_file); %#ok<LOAD>
        end
        
        builders = bs(relative_builder_idxs);
        if debug_mode(6)
            % Print the absolute builders
            disp(['Unpacking builders: ', ...
                num2str(reshape(relative_builder_idxs,1, []) + ...
                                      (time_division_idx - 1) * ...
                                      n_builders_per_time_div)]);
        end
    elseif flow_control == 0
        % use all builders to compute the channel. 
        builders = original_builder;
        if n_freq > 1
            builders = split_multi_freq(builders);
        end
        % The necessary parameters were generated beforehand.
    end
    
    if debug_mode(6)
        disp(['Generating Channel Coefficients...']);
    end
    
    
    % Actual channel computation
    c = builders.get_channels;
    
    %c is [n_RX, n_TX, n_FREQ] (but is distributed sequentially)
    % each coeff matrix is [n_rx_ele, n_tx_ele, paths, snapshot]
    %freq_response is [ Rx-Antenna , Tx-Antenna , Carrier-Index , Snapshot]
                        %element     %element
    % FULL FREQUENCY RESPONSE will be: 
    %[RX, TX, FREQ, RX_antennas, TX_antennas, carriers, snapshots]

    % NOTE: that if the number of antenna elements or carriers change from
    %frequency to frequency or from receiver to receiver, the last elements
    %will be empty. Hence, careful when selecting them is essential.
    
    
    if flow_control == 3
        % Save the channel (object) in time domain
        for i = relative_builder_idxs
            if save_partial_channel_time_domain
                c_aux = c(relative_builder_idxs == i);
                builder_idx = i + (time_division_idx - 1) * ...
                                  n_builders_per_time_div;
                save([channel_folder, 'c_', num2str(builder_idx)], ...
                     'c_aux', '-v7.3')
            end
        end
    end
    
    % The normal channel is given in (1 x length) instead in a proper
    % accessible matrix. Here it's reshaped before saved.
    if flow_control == 0
        if debug_mode(6)
            disp('Saving channel in time domain');
        end
        c = reshape(c, base_channel_dimensions);
        save([mother_folder, 'c_full'], 'c', '-v7.3');
    end
    
    % Compute frequency response right away
    % (this routine is repeated afterwards if blocking is enabled)
    if get_full_freq_response
        if flow_control == 0
            builder_idxs = [];
            time_division_idx = 0;
            name_to_save = 'fr_full';
        else
            name_to_save = 'fr_part';
            builder_idxs = relative_builder_idxs + ...
                           + (time_division_idx - 1) ...
                             * n_builders_per_time_div;
        end
        
        if debug_mode(6)
            disp(['Converting to Frequency Domain']);
        end
        
        compute_freq_response(debug_mode(6), flow_control, numerology, ...
                bandwidth, n_prb, f_values, l, c, mother_folder, ...
                channel_folder, builder_idxs, save_freq_response, ...
                n_rx, n_tx, n_freq, name_to_save);
    end
end


%% Apply the blocking (if enabled)

if flow_control == 0 || flow_control == 4
    
    if flow_control == 4
        % Load vars file
        load([input_filename, 'vars.mat']); %#ok<LOAD> 
    end
    
    % Because the channels may get quite big, it's best to do apply
    % blocking incrementally, that way parallelising is also possible

    % Blocking Model
    if apply_blocking
        
        if flow_control == 4
            
            if instance_info(2) == 0
                % Convert all of them at once
                builder_idxs = 1:n_total_builders;
            else
                % Convert the builders that were associated with a given 
                % instance
                [time_division_idx, relative_builder_idxs] = ...
                get_builder_idxs_from_instance(...
                    instance_info(2), n_instances_per_time_division, ...
                    n_builders_per_time_div);


                builder_idxs = relative_builder_idxs + ...
                                                (time_division_idx - 1) ...
                                                 * n_builders_per_time_div;
            end
            
            for builder_idx = builder_idxs
                % Load each channel part
                c = load([channel_folder, 'c_', ...
                          num2str(builder_idx), '.mat']).c_aux;

                % ------------------------------------------------------- %
                % Input variable: c 
                % C(RX1, TX1, FREQ1) = channel object.
                % size(c.coeff) = ...
                %           [ AE in RX1, AE in TX1, number paths, snapshot]

                % SANDRA'S MAGIC HERE!
                c_blocked = blocking_model(c, 1);

                % Output variable: c_blocked
                % ------------------------------------------------------- %
            
            
                % Save channel after blocking
                save([channel_blocked_folder, 'c_blocked_', ...
                      num2str(builder_idx), '.mat'], 'c_blocked', '-v7.3');

                % Compute frequency response & save to ch_blocked_folder
                if get_full_freq_response
                    if debug_mode(6)
                        disp(['Converting from Time Domain to ', ...
                              'Frequency Domain blocked channel ', ...
                              num2str(builder_idx), ' out of ', ...
                              num2str(n_total_builders)]);
                    end
                    
                    compute_freq_response(debug_mode(6), flow_control, ...
                        numerology, bandwidth, n_prb, f_values, l, ...
                        c_blocked, mother_folder, channel_blocked_folder, ...
                        builder_idx, save_freq_response, ...
                        n_rx, n_tx, n_freq, 'fr_blocked_part');
                end
            end
        end
        
        if flow_control == 0
            % The channel is computed and aggregated already, we only
            % need to iterate over the parts of the channel
            
            if debug_mode(6)
                disp('Applying Blocking!');
            end
            
            for i = 1:numel(c)
                if debug_mode(6)
                    disp(['Apply blocking to ch ', num2str(i), ...
                          ' out of ', num2str(numel(c))]);
                end
                c_blocked(i) = blocking_model(c(i), 1);
            end
            
            c_blocked = reshape(c_blocked, base_channel_dimensions);
            
            % Save channel after blocking
            save([mother_folder, 'c_blocked_full'], 'c_blocked', '-v7.3');

            % Compute frequency response 
            if get_full_freq_response
                if debug_mode(6)
                    disp(['Converting to Frequency Domain ', ...
                          '(after blocking!)']);
                end
                
                compute_freq_response(debug_mode(6), flow_control, ...
                    numerology, bandwidth, n_prb, f_values, l, ...
                    c_blocked, mother_folder, channel_blocked_folder, ...
                    builder_idxs, save_freq_response, ...
                    n_rx, n_tx, n_freq, 'fr_blocked_full');
            end
        end
    end
end
    
% Channel aggregator
if flow_control == 5
    
    % As usual, load simulation vars.
    load([input_filename, 'vars.mat']); %#ok<LOAD> 
    
    % Here are the saving options:
    % 1- Time Domain
    %    a) Leave the parts separated;
    %    b) Aggregate them into one channel file only;
    % 2- Freq Domain
    %    a) Leave the parts separated;
    %    b) Aggregate them into one coeff matrix only, for each
    %       numerology
    %
    % These choices can be made for the non-blocked channel, or for
    % the blocked channel, based on the index of the variables:
    % aggregate_full_channel_time_domain
    % &
    % aggregate_coeffs_in_freq

    % Select non-blocked or blocked channels:
    for channel_selector = 1:2
        
        if channel_selector == 1
            % Load from non-blocked channel folder
            ch_to_load = channel_folder;
            additive_to_name = '';
        else 
            % Load from blocked channel folder
            if apply_blocking == 0
                break;
            else
                ch_to_load = channel_blocked_folder;
                additive_to_name = '_blocked';
            end
        end

        
        % 1- Time Domain
        % Load all channel parts and create a c_full
        if aggregate_full_channel_time_domain(channel_selector)
            if debug_mode(6)
                if channel_selector == 1
                    disp(['Aggregating channel in Time Domain: ', ...
                          'Non-Blocked channel.']);
                else
                    disp(['Aggregating channel in Time Domain: ', ...
                          'Blocked channel.']);
                end
            end
            % load channel parts
            c = [];
            for i = 1:n_total_builders
                if debug_mode(6)
                    disp(['Loading ', num2str(i), ' out of ', ...
                          num2str(n_total_builders)]);
                end
                name_to_load = [ch_to_load, 'c', additive_to_name, '_', ...
                                num2str(i), '.mat'];
                if channel_selector == 1
                    c = [c load(name_to_load).c_aux]; %#ok<AGROW>
                else
                    c = [c load(name_to_load).c_blocked]; %#ok<AGROW>
                end
            end

            % Stitch channels of different time divisions

            % the par parts of the channel is the same for different 
            % divisions, the only difference is the path gains, which
            % depend on the coeffs
            
            % For each builder, fetch all other divisions and join them.
            c_stitched = [];
            for i = 1:n_builders_per_time_div
                if debug_mode(6)
                    disp(['Stitching ', num2str(i), ' out of ', ...
                          num2str(n_builders_per_time_div)]);
                end
                
                c_stitched = [c_stitched stitch_channels(...
                   c(1:n_builders_per_time_div:n_total_builders))]; %#ok<AGROW>
            end
            c_full = reshape(c_stitched, base_channel_dimensions);
            if debug_mode(6)
                disp('Saving channel...');
            end
            save([mother_folder, 'c', additive_to_name, '_full'],...
                 'c_full', '-v7.3');
            if debug_mode(6)
                disp('Full channel saved.');
            end
        end
        
        % 2- Frequency Domain

        % Load all frequency coefficients parts into only one matrix
        % (per numerology!)
        if aggregate_coeffs_in_freq(channel_selector)
            if debug_mode(6)
                if channel_selector == 1
                    disp(['Aggregating channel in Frequency Domain ', ...
                          'for non-blocked channel.']);
                else
                    disp(['Aggregating channel in Frequency Domain ', ...
                          'for Blocked channel.']);
                end
            end
            % Load previous ch coeffs.
            % If the purpose is to do aggregation, then all the channel
            % coeffs should be computed already.
            c = [];
            for numerology_idx = 1:length(numerology)
                for builder_idx = 1:n_total_builders
                    if debug_mode(6)
                        disp(['For Num ', ...
                              num2str(numerology(numerology_idx)), ...
                              ': aggregating frequency response ', ...
                              num2str(builder_idx), ' out of ', ...
                              num2str(n_total_builders)]);
                    end
                    
                    
                    % C(RX1, TX1, FREQ1) = 
                    %   [ AE in RX1, AE in TX1, number paths, snapshot]
                    [time_div, rx_idx, tx_idx, freq_idx] = ...
                        get_time_div_rx_tx_freq_from_builder_idx(...
                                      builder_idx, n_rx, n_tx, n_freq);

                    if bandwidth(freq_idx, numerology_idx) == 0
                        continue;
                    end
                    num = numerology(numerology_idx);



                    % The number of subcarriers makes the size of the 
                    % matrix to read change...
                    lower_trim_idx = trim_idx(time_div);
                    n_tti = trim_idx(time_div+1) - lower_trim_idx;
                    if time_div == 1
                        n_tti = n_tti + 1;
                    else
                        lower_trim_idx = lower_trim_idx + 1;
                    end

                    mat_size = [rx_ant_numel(freq_idx, rx_idx), ...
                                tx_ant_numel(freq_idx, tx_idx),...
                                n_prb(freq_idx, numerology_idx), ...
                                n_tti];

                    reading_filename = [ch_to_load, ...
                                        'fr', ...
                                        additive_to_name, '_', ...
                                        'part_', ...
                                        num2str(builder_idx), ...
                                        '_num', ...
                                        num2str(num)];

                    c_read = read_complex(reading_filename, mat_size);

                    c(rx_idx, tx_idx, freq_idx, ...
                        1:mat_size(1), 1:mat_size(2), 1:mat_size(3), ...
                        lower_trim_idx:trim_idx(time_div+1)) = c_read; %#ok<AGROW>
                end
                
                if debug_mode(6)
                    disp(['Done Appending phase. Saving full matrix...']);
                end
                
                % Save aggregated coeffs
                save_complex([mother_folder, ...
                             'fr', additive_to_name, '_full_num', ...
                              num2str(num)], c)
            end
        end
    end
end


%%

if flow_control == 0 && any(calc_stats)
    if debug_mode(6)
        disp(['Calculating Stats']);
    end
    
    % PER USER AND PER FREQUENCY                  
    % 1- Computes STD of RSS and of power in LoS
    % 2- Sees the 3 paths with biggest power and how much percentage of the
    % RSS these paths contain.
    if calc_stats(1)
        start_idx = 12;
        calc_stats1(c, diff_orthogonal_polarisation, n_rx, n_freq, ...
                    rx_mvnt, f_values, update_rate, start_idx, ...
                    variability_check, visualize_plots, manual_LOS);
    end
    if calc_stats(2)
        calc_stats2(c, n_rx, n_freq, max_num_segments, f_values, ...
                    segment_info, debug_mode)
    end

end
toc;
disp('Done.');
end