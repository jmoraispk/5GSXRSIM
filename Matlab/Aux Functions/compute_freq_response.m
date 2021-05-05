function [] = compute_freq_response(debug_var, flow_control, numerology, ...
            bandwidth, n_prb, f_values, l, c, mother_folder, channel_folder, ...
            builder_idxs, save_freq_response, n_rx, n_tx, n_freq, ...
            name_to_save)
        
    
    % Computes the frequency response for the given channels
    
    
    % if Parallel run (flow_control != 0), the saved builders are saved
    % with the absolute builder indices
        
        
    last_snapshot = size(c(1).coeff, 4);
    max_num = max(numerology);
    
    if flow_control == 0
        % then all frequencies will be in the channel, so the loop is
        % over all these frequencies
        for n = 1:length(numerology)
            if debug_var
                disp(['Numerology ', num2str(numerology(n))]);
            end
            bw = bandwidth(:, n);
            f_idxs = find(bw ~= 0);
            if isempty(find(f_idxs,1))
                disp('Empty bandwidth. Skipping.');
                continue;
            end
            prb = n_prb(:, n);
            
            
            max_ele_rx = get_max_num_elements(l.rx_array);
            max_ele_tx = get_max_num_elements(l.tx_array);
            
            freqresp = zeros([n_freq, n_tx, n_rx, ...
                              max_ele_rx, max_ele_tx, ...
                              max(prb), last_snapshot]);
            
            t_before = clock();
            for f_idx = f_idxs' %needs transpose for proper loop                    
                for tx = 1:n_tx
                    for k = 1:n_rx
                        tx_elements = l.tx_array(f_idx, tx).no_elements;
                        rx_elements = l.rx_array(f_idx, k).no_elements;

                        % We only want the necessary frequencies 
                        %outputed in each numerology
                        f_idx_on_this_num = find(f_idxs == f_idx);
                        
                        t_before_partial = clock();
                        
                        freqresp(f_idx_on_this_num, tx, k, ...
                                 1:rx_elements, 1:tx_elements, ...
                                 1:prb(f_idx), 1:last_snapshot) = ...
                                 c(k, tx, f_idx).fr(bw(f_idx), prb(f_idx)); %#ok<FNDSB>
                        
                        t_after_partial = clock();
                        t_passed = ...
                            duration(0, 0, etime(t_after_partial, ...
                                                 t_before_partial));
                        disp(['Frequency Response for (freq, tx, rx)= ', ...
                              num2str([f_idx, tx, k]), ' took ', ...
                              dur_to_str(t_passed), ' to complete']);
                    end
                end
            end
            
            t_after = clock();
            t_passed = duration(0, 0, etime(t_after, t_before));
            disp(['TOTAL TIME for this numerology Frequency Response ', ...
                  'Computation was: ', dur_to_str(t_passed), ...
                  ' to complete']);
            
            % For each numerology, save all frequencies that use it
            if save_freq_response
                if (2 ^ (max_num-numerology(n))) > 0
                    if debug_var
                        disp(['Fixing excessive TTIs for lower ', ... 
                              'numerologies']);
                    end
                    t_before = clock();
                    
                    freqresp = compress_by_averaging_every_x(...
                                           freqresp, 7, ...
                                           2 ^ (max_num-numerology(n)));        
                    
                    t_after = clock();
                    t_passed = duration(0, 0, etime(t_after, t_before));
                    disp(['Frequency compression took: ', ...
                              dur_to_str(t_passed), ' to complete']);
                end
                disp('I''m about to save a freqresp of size:');
                disp(size(freqresp));
                
                if debug_var
                    disp('Saving...');
                end
                
                save_name = [mother_folder, name_to_save, '_num', ...
                             num2str(numerology(n))];
                
                t_before = clock();
                
                save_complex(save_name, freqresp)
                
                t_after = clock(); % MODIFY FOR THE OTHERS AS WELL!
                t_passed = duration(0, 0, etime(t_after, t_before));
                disp(['Frequency saving took: ', ...   
                      dur_to_str(t_passed), ' to complete']);
                
                if debug_var
                    disp(['Coefficients saved as: ', save_name]);
                end
                
                
            end
        end
    else
        % When paralllelising, the frequency response to save will be
        % relative to each of the builders. Therefore, even if one 
        % instance computes 4 builders (because it's not on RX level, 
        % or else it would compute 1 only) each builder's response
        % will be computed and saved individually and reassembled in
        % Python


        if c(1).center_frequency ~= c(end).center_frequency
            pool_of_freqs_idx = 1:n_freq; %#ok<NASGU>
        else
            pool_of_freq_idx = find(f_values == c(1).center_frequency);
        end

        % Depending on the parallelisation methodology, one can save 
        % many numerologies for 1 or many frequencies.

        for n = 1:length(numerology)
    %        [~, n_list] = min(numerology); % there'll be only one element.
    %        for n = n_list  % It's also possible to compute only
        % for the numerology that has more samples in frequency and 
        % then compute averages for the others.
        
            for f_idx = pool_of_freq_idx'
                bw = bandwidth(:, n);
                prb = n_prb(:, n);
                if bw(f_idx) == 0
                    % This frequency doesn't have this numerology
                    continue;
                end
                for i = 1:length(c)
                    if c(i).center_frequency ~= f_values(f_idx)
                        % this channel doesn't have this frequency
                        continue;
                    end
                    
                    t_before = clock();
                    
                    freqresp = c(i).fr(bw(f_idx), prb(f_idx));
                    
                    t_after = clock();
                    t_passed = duration(0, 0, etime(t_after, t_before));
                    disp(['Frequency Response took: ', ...
                          dur_to_str(t_passed), ...
                          ' to complete']);
                    
                    if save_freq_response
                        if debug_var
                            disp(['Saving freq response part ', ...
                              num2str(i), ' out of ', ...
                              num2str(length(c)), ...
                              ' for num ', num2str(numerology(n))]);
                        end

                        % Save the coefficients in binary format


                        save_name = [channel_folder, name_to_save, '_', ...
                            num2str(builder_idxs(i)), ...
                            '_num', num2str(numerology(n))];
                        
                        if (2 ^ (max_num-numerology(n))) > 0
                            if debug_var
                                disp(['Fixing excessive TTIs for lower ', ... 
                                      'numerologies']);
                            end
                            t_before_partial = clock();
                            
                            freqresp = compress_by_averaging_every_x(...
                                           freqresp, 4, ...
                                           2 ^ (max_num-numerology(n)));
                            
                            t_after_partial = clock();
                            t_passed = ...
                                duration(0, 0, etime(t_after_partial, ...
                                                     t_before_partial));
                            disp(['Frequency compression took: ', ...   
                              dur_to_str(t_passed), ' to complete']);
                        end
                        
                        disp('I''m about to save a freqresp of size:');
                        disp(size(freqresp));
                
                        
                        % This commented save was to a compressed .mat
                        % file, which takes considerably longer to save.
                        %  save(saved_name, 'freqresp', '-v7.3');
                        t_before = clock();
                        
                        % This is the binary save:
                        save_complex(save_name, freqresp);
                        
                        if debug_var
                            disp(['Coefficients ', ...
                                    ' save as ', save_name]);
                        end
                        
                        t_after = clock();
                        t_passed = duration(0, 0, etime(t_after, t_before));
                        disp(['Frequency saving took: ', ...   
                              dur_to_str(t_passed), ' to complete']);
                    end
                end
            end
        end
    end
end