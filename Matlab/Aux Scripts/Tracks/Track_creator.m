% The purpose of this file is to create tracks for certain seeds, such that
% different speeds can go over those tracks multiple times, hence making
% speeds comparable.

% IMPORTANT NOTE: we are using a few variables to compute these tracks and
%                 the tracks will depend on those variables, such as 
%                 position of the users. Therefore, even though the file
%                 name of the track won't change, you need to be careful
%                 not to reuse tracks when the average positions of the 
%                 users had changed.

% INPUT THE CORRECT PATH to the vars.mat of choice
load('vars_4ues.mat'); 
% About the vars.mat: barely any information is needed from the vars.mat
% file. A normal simulation with that number of users with those positions
% is enough. No need to generate the whole thing, we just need the vars.mat
% which comes from the setup right away.

% useful things to control for testing:
% turn_off_positions = 1;

% THINGS WE CAN CHANGE:
%   - SEED
%   - mvnt_values
% speaking_time = 4;
mvnt_vals = 4;
SEEDs = [1:5];

for SEED = SEEDs
    % SEED is set just in the beginning of the first set of tracks
    rng(SEED, 'twister');

    
    % Now, we create the content of those tracks, for each speed.

    % Step list:
    % 1 - create macro positions and macro orientations for the lowest speed
    % 2 - Loop for each speed:
    %      2.1 - interpolate macro positions
    %      2.2 - manually interpolate orientations
    %      2.3 - Save layout for this speed

    % Assumptions:
    %    General:
    %    - all users have the same speed
    %    - there are no static UEs
    % 
    %    Orientation:
    %    - The speaker list done already.
    %    - The speaker list is well created, i.e. through the automatic 
    %      process in the main Meeting (includes having a speaker record 
    %      that is longer than the simulation duration);
    %    - The speaker list does not include the physical users, only the
    %      virtual
    %    - The head_model_rotation it is either 0 (for cameras) or 5 (for
    %    users)
    %    - Every user has the same speaking time (so we don't measure from 
    %      the speaking list how much that is for each user, but we see it 
    %      from the variable used to create the speaking list, speaking time.

    lowest_mvnt_value = min(mvnt_vals);

    l_lowest_speed = qd_layout();
    % Position Model

    rx_mvnt = repmat(lowest_mvnt_value, [1 n_room(1)]);
    rx_mvnt = [rx_mvnt zeros([1, size(cam_pos, 2)])];

    [~, rot_intervals] = get_speed_and_rot_int(rx_mvnt);
    slowest_rot_int = min(rot_intervals); % all equal assumption!

    % Generate a certain number of tracks for that mvnt val, one per UE
    x_pos_macro = [];
    y_pos_macro = [];
    z_pos_macro = [];

    x_rot_macro = [];
    y_rot_macro = [];
    z_rot_macro = [];

    % x/y/z_pos/rot_macro_aux are the auxiliar vectors for a given user

    for k = 1:n_rx
        % %%%%%%%%%%%%%%% Positions %%%%%%%%%%%%%%%%%%%%
        % Number of macro positions needed
        n_macro_pos = ceil(n_mvnt_samples_aux * rx_mvnt(k));

        % Sphere/Ellipsoid movement.
        x_pos_macro_aux = randn(1,n_macro_pos) .* sigma_head; %#ok<*SAGROW>
        y_pos_macro_aux = randn(1,n_macro_pos) .* sigma_head; 
        z_pos_macro_aux = randn(1,n_macro_pos) .* sigma_head/2; 

        % %%%%%%%%%%%%%% Orientations %%%%%%%%%%%%%%%%%%

        % Number of macro orientations needed per speaking interval
        n_macro_ori = ceil( speaking_time / rot_intervals(k) );

        % Go to each speaking interval and sample the macro positions there

        %for safety
        x_rot_macro_aux = [];
        y_rot_macro_aux = [];
        z_rot_macro_aux = [];
        rot_instants = 0;

        % Save n_macro_ori for each speaker orientation
        for i = 1:(size(speaker_list,1)-1)
            % Calc angle of each speaker (average orientation)
            ang = calc_2Dangle_from_pos(rx_pos(:,k), ...
                                        all_users_pos(:,speaker_list(i,2)));   

            % We assume each phy user can never speak. 
            % If that assumption is broken, then stop execution.
            if speaker_list(i,2) == k 
                % We assume only the virtual users speak, so it never 
                % happens the case where the user we are moving ends up 
                % speaking, because in that scenario, we would have to make
                % him look somewhere else random, and we don't want that 
                % extra piece of randomness in the tracks.
                disp(['If the speaker list is well-made, you should ', ...
                      'never see this message']);
                return;
            end

            %#ok<*AGROW> - suppress size change: couldn't avoid

            % Create wobbling directions on that angle
            for j = 1:n_macro_ori
                x_rot_macro_aux = [x_rot_macro_aux ...
                                   uniform(-bank_lim/2, bank_lim/2, [1 1])]; 
                y_rot_macro_aux = [y_rot_macro_aux ...
                                   uniform(-tilt_lim/2, tilt_lim/2, [1 1])];
                z_rot_macro_aux = [z_rot_macro_aux ...
                                   uniform(ang - fdang, ang + fdang, [1 1])];

                if j == n_macro_ori
                    rot_instants = [rot_instants speaker_list(i+1,1)];
                else
                    rot_instants = [rot_instants ...
                                    rot_instants(end) + rot_intervals(k)];
                end
            end
        end

        rot_instants(end) = [];
    %     disp(['k = ', num2str(k), ', rot_insts: ', newline, ...
    %                   num2str(rot_instants), newline, ...
    %                   'z_rot_macro = ', newline, num2str(z_rot_macro)]);

        % Trim the zero that was there in the beginning.
        x_pos_macro = [x_pos_macro; x_pos_macro_aux];
        y_pos_macro = [y_pos_macro; y_pos_macro_aux];
        z_pos_macro = [z_pos_macro; z_pos_macro_aux];

        x_rot_macro = [x_rot_macro; x_rot_macro_aux];
        y_rot_macro = [y_rot_macro; y_rot_macro_aux];
        z_rot_macro = [z_rot_macro; z_rot_macro_aux];
    end

    
    % Step 2

    % Define how many macro orientations exist per speaking interval
    n_ori_per_spk_int = ceil(speaking_time / rot_intervals(end));
    % These will be the number of samples we'll use circularly each
    % speaking interval, the higher the speed, the more we use those 
    % n_ori_per_spk_int samples.
    
    
    n_intervals = length(speaker_list) - 1; 

    rotation_times = (1:n_intervals) * speaking_time;

    for mvnt_val = mvnt_vals

        disp(['Generating tracks for SEED ', num2str(SEED), ...
              ', SPEED ', num2str(mvnt_val), '...']);
        
        % Create movement vectors from the speeds to simulate
        rx_mvnt = repmat(mvnt_val, [1 n_room(1)]);
        rx_mvnt = [rx_mvnt zeros([1, size(cam_pos, 2)])];

        [mvmnt_speeds, rot_intervals] = get_speed_and_rot_int(rx_mvnt);

        l_aux = qd_layout();
        l_aux.no_rx = n_rx;

        for k = 1:n_rx
    %         disp_num('k',k);

            % Create tracks
            l_aux.rx_track(1, k) = qd_track('linear', 1, 0);
            l_aux.rx_track(1, k).name = ['rx-track', num2str(k)];
            l_aux.rx_track(1, k).initial_position = rx_pos(:,k);

            % Replicate macro positions (for higher speeds)
            mult = round(mvnt_val / lowest_mvnt_value);
            x_aux = repmat(x_pos_macro(k,:), [1 mult]);
            y_aux = repmat(y_pos_macro(k,:), [1 mult]);
            z_aux = repmat(z_pos_macro(k,:), [1 mult]);

            % Assign macro positions
            l_aux.rx_track(1, k).positions = [x_aux; y_aux; z_aux];

            % Interpolate positions (samples per metre)
            % (& orientations, but these will be overwritten)
            l_aux.rx_track(1,k).interpolate_positions(round( 1 /...
                                       (mvmnt_speeds(k) * update_rate)));

            % Trim excessive positions
            l_aux.rx_track(1,k).positions = l.rx_track(1,k).positions(:, ...
                      begin_slack:(sim_duration_snapshot + ...
                                   begin_slack - 1 + extra_sample));

            if turn_off_positions
                l_aux.rx_track(1,k).positions = ...
                    zeros(3, sim_duration_snapshot + extra_sample); 
            end

            if turn_off_orientations
                continue;
                % orientations are 0 by default
            end


            % Create orientations and interpolate them manually:
            x_rot = [];
            y_rot = [];
            z_rot = [];

            % There are [is] interval samples in a rotation interval...
            is = round(rot_intervals(k) / update_rate);

            %disp_num('is', is);
            % Let us hope this works well for all update rates (which vary 
            % with the time compression variable)

            % prepare how many times the macro orientations are going to be
            % repeated
            sample_idxs_reps = ceil(slowest_rot_int / rot_intervals(k)) + 1;

            % loop over each interval a speaker speaks
            for int = 1:n_intervals
                % Which macro samples have the orientation of this interval
                sample_idxs = (int-1)*n_ori_per_spk_int+1:...
                              (int) * n_ori_per_spk_int;
                flipped_idxs = flip(sample_idxs);

                % Make a big enough list with sample idxs
                sample_idxs = [sample_idxs(1) ...
                       repmat([sample_idxs(2:end) flipped_idxs(2:end)], ...
                              [1 sample_idxs_reps])];

                % We'll be drawing from these indices until we have reached
                % the point of switching interval, which will take more 
                % samples for higher speeds.

                % In the second speaking interval and from that point 
                % onwards, we don't use the first macro orientation because
                % it is replaced by whatever orientation we have previously,
                % to provide continuity. This is done in two steps:
                if int ~= 1
                    % 1- Remove all occurences of the first sample
                    sample_idxs = sample_idxs(sample_idxs ~= sample_idxs(1));
                    % 2- Remove consecutive repeated indices
                    sample_idxs(diff(sample_idxs)==0) = [];
                end

                %disp_num('sample_idxs', sample_idxs);
    %             disp(['int: ', num2str(int)]);

                % Manual interpolation
                % get the snapshot of start
                idx2 = (int-1) * speaking_time / update_rate;
                for i = 1:length(sample_idxs)
                    idx1 = idx2 + 1;
                    idx2 = idx2 + is;

%                     disp_num('idx1', idx1);
%                     disp_num('idx2', idx2);

                    macro_idx1 = sample_idxs(i);
                    % if, by any chance, the sample_idxs are not enough, 
                    % then create more of them.
                    if i+1 > length(sample_idxs)
                        error(['The sample list was not long enough for ', ...
                               'speed ', num2str(rx_mvnt(1))]);
                    end
                    macro_idx2 = sample_idxs(i+1);

%                     disp(['i: ', num2str(i)]);
%                     disp(['Macro idx: ', num2str(macro_idx1)]);

                    % IMPORTANT: don't forget to copy the last angle in x_rot!
                    %            That way we'll have continuity. This is only
                    %            necessary after the first speaking interval

                    x_first_rot_macro_sample = x_rot_macro(k, macro_idx1);
                    y_first_rot_macro_sample = y_rot_macro(k, macro_idx1);
                    z_first_rot_macro_sample = z_rot_macro(k, macro_idx1);
                    if i == 1 && int ~= 1
                        %disp('copied first sample');

                        % CHECK THIS SHIT OUT
                        % PREVIOUSLY WE were overriding the x_rot_macro.

                        x_first_rot_macro_sample = x_rot(end);
                        y_first_rot_macro_sample = y_rot(end);
                        z_first_rot_macro_sample = z_rot(end);
                    end

                    x_rot(idx1:idx2) = linspace(x_first_rot_macro_sample, ...
                                                x_rot_macro(k, macro_idx2), is);
                    y_rot(idx1:idx2) = linspace(y_first_rot_macro_sample, ...
                                                y_rot_macro(k, macro_idx2), is);

                    % The interpolation on the azimuth needs to be a tad 
                    % more careful, because we need to interpolate in front
                    % of the table, around the outside!
                    z_rot1 = z_first_rot_macro_sample;
                    z_rot2 = z_rot_macro(k, macro_idx2);
%                     disp_num('z_rot1', z_rot1);
%                     disp_num('z_rot2', z_rot2);

                    
                    % INTERPOLATE FUNCTIONS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    
                    z_rot(idx1:idx2) = ...
                        interpolate_orientation_respecting_table(z_rot1, ...
                                                                 z_rot2, is);

                    % If this is supposed to be the last interpolation, 
                    % then trim to fit the speaking time and stop the loop
                    threshold_1 = rotation_times(int) / update_rate;
                    threshold_2 = length(l_aux.rx_track(1,k).orientation) - 1;
                    if idx2 >= threshold_1 || idx2 >= threshold_2
                        if idx2 >= threshold_1
                            new_idx2 = threshold_1;
                        else
                            new_idx2 = threshold_2;
                        end
                        if int == n_intervals
                            % the extra sample has to do with time interp. 
                            new_idx2 = new_idx2 + extra_sample;
                        end
                        x_rot(new_idx2+1:idx2) = [];
                        y_rot(new_idx2+1:idx2) = [];
                        z_rot(new_idx2+1:idx2) = [];
                        break;
                    end
                end
            end

            if size(l_aux.rx_track(1,k).orientation, 2) ~= length(z_rot)
                % This can happen for two reasons:
                % 1- the rotation intervals of a given speed is a divisor
                %    of the speaking time. E.g. 0.5s rotation interval and 
                %    4s speaking time. The issue is that there won't be any
                %    extra sample to add at the end. So, we just replicate 
                %    the last one, because it will not make any difference 
                %    at all.
                x_rot(end+1) = x_rot(end);
                y_rot(end+1) = y_rot(end);
                z_rot(end+1) = z_rot(end);

                % 2- we had some mistake... And in this is the case, the
                % mistake will most likely persist in an obvious way and 
                % raise an error in the next step.
            end
%             disp(size(x_rot));
            l_aux.rx_track(1,k).orientation = [x_rot; y_rot; z_rot];
            
            % Last small thing to complete the tracks:
            l_aux.rx_track(1,k).segment_index = segment_info(k).segment_index;
            l_aux.rx_track(1,k).scenario = segment_info(k).scenario;
        end

        % Save l_aux for this speed.

        filename = ['Tracks\Track', ...
                    '_SEED', num2str(SEED), ...
                    '_SPEED', num2str(mvnt_val), ...
                    '_UE', num2str(n_rx), '.mat'];

        save(filename, 'l_aux');
    end
end

disp('Track generation completed.');

