function [speaker_list] = head_model(l, k, mvmnt_speeds, rot_intervals, ...
                                     update_rate, bank_lim, tilt_lim, ...
                                     fdang, all_users_pos, rx_pos, ...
                                     n_users, speaker_list, ...
                                     staring_param, speaking_avg_time, ...
                                     debug_mode, ...
                                     head_model_rotation, ...
                                     sim_duration_snapshot,...
                                     simulation_duration, extra_sample)

    % The names of the variables of this function are exactly the same
    % as in the main file. k represents the k-th user (user with index k).
    


    %For head models with time intervals, interpolation is done manually
    if head_model_rotation(k) > 1
        %Random with rotation intervals

        %at speed sp[m/s], update rate ur[s/sample], the amount of 
        %snapshots in a rotation interval [s] is interval_samples (is)
        is = round(mvmnt_speeds(k) * rot_intervals(k) / update_rate);
        n_ang = ceil(sim_duration_snapshot / is) + 1;
        if head_model_rotation(k) == 2
            ang_list = [];
            for n = 1:n_users
                if n ~=k
                    ang_list = ...
                        [ang_list calc_2Dangle_from_pos(rx_pos(:,k), ...
                         all_users_pos(:,n))];   
                end
            end
            azi_lim_1 = min(ang_list);
            azi_lim_2 = max(ang_list);

            
            x_rot_aux = uniform(-bank_lim, bank_lim, [1 n_ang]);
            y_rot_aux = uniform(-tilt_lim, tilt_lim, [1 n_ang]);
            z_rot_aux = uniform(azi_lim_1, azi_lim_2, [1 n_ang]);
        elseif head_model_rotation(k) < 5
            %when looking at someone/something, reduce oscillations
            x_rot_aux = uniform(-bank_lim/2, bank_lim/2, [1 n_ang]);
            y_rot_aux = uniform(-tilt_lim/2, tilt_lim/2, [1 n_ang]);

            %randomize people and look at them
            z_rot_aux = zeros([1 n_ang]);
            %be sure, self is not chosen
            pool_usr = linspace(1, n_users, n_users);
            pool_usr(k) = [];
                for i = 1:n_ang
                    if head_model_rotation(k) == 4
                        %staring probability
                        if i > 1
                            staring_choice = randi(staring_param);
                            if staring_choice == 1 %roll users again
                                usr_pick = pool_usr(randi(n_users-1));
                            end %else keep the same user
                        else
                            usr_pick = pool_usr(randi(n_users-1));
                        end
                    else
                        usr_pick = pool_usr(randi(n_users-1));
                    end
                    ang = calc_2Dangle_from_pos(rx_pos(:,k), ...
                               all_users_pos(:,usr_pick));

                    z_rot_aux(i) = uniform(ang - fdang, ...
                                           ang + fdang, [1 1]);
                end
        end

        if head_model_rotation(k) < 5
            for i = 1:(n_ang-1)
                idx1 = (i-1)*is + 1;
                idx2 = i*is;
                x_rot(idx1:idx2) = linspace(x_rot_aux(i), x_rot_aux(i+1), is);
                y_rot(idx1:idx2) = linspace(y_rot_aux(i), y_rot_aux(i+1), is);
                z_rot(idx1:idx2) = linspace(z_rot_aux(i), z_rot_aux(i+1), is);
                if debug_mode(1)
                    disp(['k = ', num2str(k), ', is: ', num2str(is),...
                        ', idx1 = ', num2str(idx1), ...
                        ', idx2 = ', num2str(idx2), 'size(x_rot) = ',...
                        num2str(size(x_rot))]);
                end
            end
        else %head_model_rotation(k) = 5
            %Model "look at the speaker"
            %speaker_list has [time_of_speach_beginning speaker; 
                              %time_of_speach_beginning speaker;...
            %when other speaker starts, there's a transition (head
            %rotation). While one speaker is speaking, there are 
            %minor tilts and rotations
            if isempty(speaker_list)
                %then randomize the speakers and speaking times
                pool_usr = round(linspace(1, n_users, n_users));
                speaker_list(1,1) = 0;
                speaker_list(1,2) = pool_usr(randi(n_users-1));
                % to make (even) less likely to continue talking
                staring_cnt = 0; 
                while speaker_list(end,1) < simulation_duration
                    %keep starring for a couple of seconds more or
                    %change focus

                    %the person keeps talking
                    if randi(staring_param + staring_cnt) == 1 
                        speaker_list(end+1,2) = speaker_list(end, 2);
                        speaker_list(end,1) = speaker_list(end-1,1) + ...
                            uniform(speaking_avg_time*0.3, ...
                                    speaking_avg_time*0.7, [1 1]);

                        staring_cnt = staring_cnt + 1;
                    else %change focus
                        speaker_list(end+1,2) = pool_usr(randi(n_users-1)); %#ok<*SAGROW>
                        speaker_list(end,1) = speaker_list(end-1,1) + ...
                            uniform(speaking_avg_time*0.5, ...
                                    speaking_avg_time*1.5, [1 1]);

                        staring_cnt = 0;
                    end
                end
                if debug_mode(4)
                    disp(speaker_list)
                end
            end
            %Now there's a speaker order established for sure
            %we need to attribute every user the same direction

            %be sure, self is not chosen
            pool_usr = linspace(1, n_users, n_users);
            pool_usr(k) = [];

            %for safety 
            x_rot_aux = [];
            y_rot_aux = [];
            z_rot_aux = [];


            % Setup angles to be interpolated
            %rotation_instants has the timestamps of each rotation.
            for i = 1:(size(speaker_list,1)-1)
                % #directions to look when looking at one person
                ni = ceil( (speaker_list(i+1, 1) - speaker_list(i, 1)) ...
                             / rot_intervals(k)  );
                %(in order to create sort of a wandering look, to
                %simulate the normal head rotations

                % Calc the angle to look at 
                if speaker_list(i,2) == k %if I'm the one talking.
                    if head_model_rotation(k) == 6
                        %talk to the person before (HEAD MODEL 6)
                        if i == 1
                            % if it's the first user, pick at the last
                            ang = calc_2Dangle_from_pos(rx_pos(:,k), ...
                              all_users_pos(:,speaker_list(end,2)));
                        else
                            % if it's not, look at the previous oneo
                            ang = calc_2Dangle_from_pos(rx_pos(:,k), ...
                              all_users_pos(:,speaker_list(i-1,2)));
                        end
                        
                    else
                        %talk to a random person
                        ang = calc_2Dangle_from_pos(rx_pos(:,k), ...
                           all_users_pos(:,pool_usr(randi(n_users-1))));
                    end
                else
                    ang = calc_2Dangle_from_pos(rx_pos(:,k), ...
                           all_users_pos(:,speaker_list(i,2)));   
                end

                if i == 1
                %So that the orientations start from the very
                %first instant, not only from rot_interval(k)
                %onwards. So, randomize an angle to start.
                    x_rot_aux = [x_rot_aux ...
                      uniform(-bank_lim/2, bank_lim/2, [1 1])]; 
                    y_rot_aux = [y_rot_aux ...
                      uniform(-tilt_lim/2, tilt_lim/2, [1 1])];
                    z_rot_aux = [z_rot_aux ...
                      uniform(ang - fdang, ang + fdang, [1 1])];
                    rot_instants = [0];
                end

                %#ok<*AGROW> - suppress size change: couldn't avoid

                %create wandering directions on that angle
                for j = 1:ni
                    x_rot_aux = [x_rot_aux ...
                              uniform(-bank_lim/2, bank_lim/2, [1 1])]; 
                    y_rot_aux = [y_rot_aux ...
                              uniform(-tilt_lim/2, tilt_lim/2, [1 1])];
                    z_rot_aux = [z_rot_aux ...
                              uniform(ang - fdang, ang + fdang, [1 1])];

                    if j == ni
                        rot_instants = [rot_instants speaker_list(i+1,1)];
                    else
                        rot_instants = [rot_instants ...
                                 rot_instants(end) + rot_intervals(k)];
                    end
                end
            end

            if debug_mode(5)
                disp(['k = ', ...
                      num2str(k), ', rot_insts: ', newline, ...
                      num2str(rot_instants), newline, ...
                     'z_rot_aux = ', newline, num2str(z_rot_aux)]);
            end

            %manual interpolation
            idx2 = 0;
            for i = 1:length(x_rot_aux)-1 
                %calc sample interval based on speaker_times
                is = round((rot_instants(i+1) - rot_instants(i)) ...
                            / update_rate); 
                idx1 = idx2 + 1;
                idx2 = idx2 + is;

                x_rot(idx1:idx2) = linspace(x_rot_aux(i), ...
                                            x_rot_aux(i+1), is);
                y_rot(idx1:idx2) = linspace(y_rot_aux(i), ...
                                            y_rot_aux(i+1), is);
                z_rot(idx1:idx2) = ...
                    interpolate_orientation_respecting_table(z_rot_aux(i), ...
                                                             z_rot_aux(i+1), is);

            end
        end
        
        
        if sim_duration_snapshot > length(x_rot)
            % this problem is fixed by assuming the last speaker is
            % speaking until the end of the simulation (by placing a 
            % speaker that only starts speaking many seconds before the
            % last, which will never be reached.
            
            
            error(['sim_duration_snapshot > length(x_rot). ', ...
                   'LIKELY PROBLEM: the list of speakers is not ', ...
                   'long enough.']);
        end
            
            
        %trim excessive snaps so size(orient.) = size(pos)
        l.rx_track(1,k).orientation = ...
                          [x_rot(1:sim_duration_snapshot+extra_sample); ...
                           y_rot(1:sim_duration_snapshot+extra_sample); ...
                           z_rot(1:sim_duration_snapshot+extra_sample)];
                       
        % PS: the extra sample has to do with time interpolation. See main.
    end
end

