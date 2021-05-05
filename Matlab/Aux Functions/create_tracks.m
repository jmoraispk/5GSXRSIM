function [] = create_tracks(l, head_model_rotation, bank_lim, tilt_lim, ...
                            n_users, sigma_head, all_users_pos, rx_pos, ...
                            r_cam, n_rx, n_mvmt_samples, rx_mvnt, ... 
                            sim_duration_snapshot, update_rate, ...
                            begin_slack, fdang, speaker_list,...
                            staring_param, speaking_avg_time, debug_mode,...
                            simulation_duration, extra_sample, ...
                            segment_info, turn_off_orientations, ...
                            turn_off_positions)

               
    [mvmnt_speeds, rot_intervals] = get_speed_and_rot_int(rx_mvnt);
         
    for k = 1:n_rx

        if mvmnt_speeds(k) == 0
            %camera
            l.rx_track(1,k) = qd_track('circular', r_cam * pi * 2, 0);
            l.rx_track(1,k).name = ['rx-track', num2str(k)];
            l.rx_track(1,k).initial_position = rx_pos(:, k);

            %some samples are lost. Add a few and trim excess at the end.
            samples_slack = 200;

            l.rx_track(1,k).interpolate_positions(...
                 (sim_duration_snapshot + samples_slack)/(r_cam * pi * 2));
            %trim excess
            l.rx_track(1,k).positions = ...
                l.rx_track(1,k).positions(:, 1:sim_duration_snapshot + ...
                                               extra_sample);
            l.rx_track(1,k).orientation = ...
                zeros(size(l.rx_track(1,k).positions));
        else
            %moving present user
            l.rx_track(1, k) = qd_track('linear', 1, 0);
            l.rx_track(1, k).name = ['rx-track', num2str(k)];
            l.rx_track(1, k).initial_position = rx_pos(:,k);
            %Position update

            %n changes with the user's amount of movement
            n = ceil(n_mvmt_samples * rx_mvnt(k));

            % Sphere/Ellipsoid movement.
            x = randn(1,n) .* sigma_head;
            y = randn(1,n) .* sigma_head;
            z = randn(1,n) .* sigma_head/2;

            l.rx_track(1, k).positions = [x; y; z];



            %if head_model_rotation = 0, it only interpolates.
            if turn_off_orientations
                head_model_rotation(k) = 0; 
            end


            % To leverage the interpolation of positions, head_model 1
            % needs to be set outside of head_model().
            % All other models implementations are inside that function

            % Head Model 1 is random and has no rotation intervals defined
            if head_model_rotation(k) == 1
                % The range of rotation (on azimuth) is based on the 
                % participants on the extremes

                ang_list = [];
                for n = 1:n_users
                    if n ~=k
                        ang_list = ...
                            [ang_list calc_2Dangle_from_pos(rx_pos(:,k), ...
                             all_users_pos(:,n))];    %#ok<AGROW>
                    end
                end
                azi_lim_1 = min(ang_list);
                azi_lim_2 = max(ang_list);

                %Random, no rotation intervals
                x_rot = uniform(-bank_lim, bank_lim, [1 n]);
                y_rot = uniform(-tilt_lim, tilt_lim, [1 n]);
                z_rot = uniform(azi_lim_1, azi_lim_2, [1 n]);
                l.rx_track(1,k).orientation = [x_rot; y_rot; z_rot];
            end
            % End of Pre-rotation preparation of head_model_1.


            %interpolate positions & orientations (samples per metre)
            l.rx_track(1,k).interpolate_positions(round( 1/...
                                       (mvmnt_speeds(k)*update_rate)));

            l.rx_track(1,k).positions = l.rx_track(1,k).positions(:, ...
                      begin_slack:(sim_duration_snapshot + ...
                                   begin_slack - 1 + extra_sample));

            if turn_off_positions
                l.rx_track(1,k).positions = zeros(3, sim_duration_snapshot); 
            end

            % Separate interpolation is required when orientations and 
            % positions operate at different time scales, i.e. if position 
            % has a speed and the rotation has a time interval, manual 
            % setup is necessary

            %Fill correct orientations:
            % Note: for head models with time intervals, 
            % interpolation is done manually
            if head_model_rotation(k) > 1 
                speaker_list = ...
                    head_model(l, k, mvmnt_speeds, rot_intervals, ...
                               update_rate, bank_lim, tilt_lim, fdang, ...
                               all_users_pos, rx_pos, n_users, speaker_list,...
                               staring_param, speaking_avg_time, debug_mode,...
                               head_model_rotation, sim_duration_snapshot, ...
                               simulation_duration, extra_sample);
            end
        end

        % After position update, update segments and scenarios
        l.rx_track(1,k).segment_index = segment_info(k).segment_index;
        l.rx_track(1,k).scenario = segment_info(k).scenario;   
    end
end