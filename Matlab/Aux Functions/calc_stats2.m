function [] = calc_stats2(c, n_rx, n_freq, max_num_segments, ...
                          f_values, segment_info,  debug_mode)
    
    % PER USER AND PER FREQUENCY                  
    % 1- Computes STD of RSS and of power in LoS
    % 2- Sees the 3 paths with biggest power and how much percentage of the
    % RSS these paths contain.
                      
    last_snapshot = size(c(1).coeff, 4);
    
    %compute max number of paths
    max_num_paths = size(c(1,1,1).coeff,3); 

    %avg pow per path per segment per user
    avg_pow_on_path = zeros(n_rx, max_num_segments, n_freq, max_num_paths);
    pow_percent_per_path = zeros(n_rx, max_num_segments, n_freq, ...
                                 max_num_paths);

    for f = 1:n_freq
        fprintf('Freq = %.1f GHz \n', round(f_values(f) / 1e9,1))
        for k = 1:n_rx
            for seg_idx = 1:length(segment_info(k).segment_index)
                % Get segment indices
                idx1 = segment_info(k).segment_index(seg_idx);
                if seg_idx == length(segment_info(k).segment_index)
                    idx2 = last_snapshot; %to the end
                else
                    idx2 = segment_info(k).segment_index(seg_idx + 1);
                end

                
                %avg power per segment
                avg_pow_on_path(k, f, seg_idx, :) = ...
                    squeeze(mean(abs(c(k,f).coeff(1,1,:,idx1:idx2)).^2,4));
                
                if debug_mode(3)
                    % Get scenario name
                    scen_name = char(segment_info(k).scenario(seg_idx));
                    scen_name = scen_name(end-3:end);
                
                    disp(['User ', num2str(k), ', Freq = ', f_str,...
                      ' Segment = ', num2str(seg_idx), ' - ', scen_name, ...
                      ', Avg power per path ', num2str(avg_pow_on_path(k, seg_idx, :))]);
                end
            end

            disp(['----------Percentages for USER ', num2str(k), '----------']);
            for seg_idx = 1:length(segment_info(k).segment_index)
                pow_sum_per_segment = sum(avg_pow_on_path(k, f, seg_idx, :));

                % Get scenario name
                scen_name = char(segment_info(k).scenario(seg_idx));
                scen_name = scen_name(end-3:end);

                pow_percent_per_path(k, f, seg_idx, :) = ...
                       round( squeeze(avg_pow_on_path(k, f, seg_idx, :)) ./ ...
                              pow_sum_per_segment * 100 , 3  );

                fprintf('Segment = %d (%s). ', seg_idx, scen_name);

                %disp([', % of power/path ', num2str(pow_percent_per_path(k, f, seg_idx, :))]);

                % Consider the best 3 paths
                take_the_best_n = 3;
                % Percentage of power in best paths
                aux = sort(squeeze(pow_percent_per_path(k, f, seg_idx, :)), 'descend');

                stat1_pows = round(aux(1:take_the_best_n))';
                stat1_sum_pow = sum(stat1_pows);
                stat1_idxs = []; %safety first
                for i = 1:take_the_best_n
                    idxs = find(squeeze(pow_percent_per_path(k, f, seg_idx, :)) == aux(i));
                    stat1_idxs(i) = idxs(1); %#ok<AGROW> %normally only happens for 0.
                end

                disp(['The ', num2str(take_the_best_n), ...
                      ' paths with most power have: ',  ...
                      num2str(stat1_sum_pow), ' % of the received power.']);


                %average power drop in (%) normalized to the total power in those paths
                avg_pow_percent_drop = round(avg_drop(stat1_pows));

                disp(['These paths idxs are: ', num2str(stat1_idxs), ...
                      '; and have: ', num2str(stat1_pows), ...
                      ' % of power, with avg % drop:', ...
                      num2str(avg_pow_percent_drop), '%.']);

                %figure;
                %histogram(aux);
            end
        end
    end

    %disp(['Reminder: Don''t forget to account for the total power', ...
    %      'contained in the 3 paths percentage while analysing ', ...
    %      'the average % drop among those paths']);
    disp(['---------------------------------------------------', newline]);
end

