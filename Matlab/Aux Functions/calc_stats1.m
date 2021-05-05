function [] = calc_stats1(c, diff_orthogonal_polarisation, n_rx, n_freq, ...
                          rx_mvnt, f_values, update_rate, start_idx, ...
                          variability_check, visualize_plots, manual_LOS)

    % Computes received power plots.
    
    last_snapshot = size(c(1).coeff, 4);
    
    for p=1:(1 + diff_orthogonal_polarisation)
        
    los_power = zeros(n_rx, n_freq, last_snapshot);
    total_accummulated_power = zeros(n_rx, n_freq, last_snapshot);

    for k=1:n_rx
        for f = 1:n_freq
            %1- sum the coefficients over all transmitter elements 
            % (note: it's a complex sum!)
            %2- sum the powers of all receiving elements 
            % (so: sum(abs(coeff).^2) )
            %3- sum the powers in every path
            %aa = c(k,f).coeff(:,:,1,:);
            %ab = sum(aa,2);              %tx contributions COMPLEX sum
            %ac = sum(abs(ab) .^2, 1); %sum of the powers in each receiver
            %pow_per_path = squeeze(ac);
            
            % If the polarisations are separted, they will be intercalated
            % in the channel response
            if diff_orthogonal_polarisation
                pow_per_path = sum(abs(sum(...
                       c(k + n_rx * (f-1)).coeff(:,p:2:end,:,:),2)) .^2,1);
            else
                pow_per_path = sum(abs(sum(...
                         c(k + n_rx * (f-1)).coeff(:,1:end,:,:),2)) .^2,1);
            end
            
            los_power(k, f, :) = squeeze(pow_per_path(1,1,1,:));
                                                            %over all paths
            total_accummulated_power(k, f, :) = sum(pow_per_path,3); 

            % Calculate the power
            pow  = 10*log10( squeeze(total_accummulated_power(k, f, :)) );    
            pow_percent_in_LoS = squeeze( los_power(k, f, :) ./ ...
                                 total_accummulated_power(k, f, :) * 100);

            %strings 
            f_str = num2str(f_values(f) / 1e9); 

            disp(['User ', num2str(k), ', Freq = ', f_str,...
                  ' RSS std = ', num2str(std(pow)), ...
                  ', % Power in LoS std = ', ...
                  num2str(std(pow_percent_in_LoS))]);
            
            % Compute maximum differences
            if variability_check
                [a, b] = max(pow);
                disp(a); disp(round(b * update_rate,2));
                for i = 1:10
                    m = max(calculate_interval_diffs(pow,i));
                    fprintf('%2.0f TTI (%1.2f ms) --> %1.2f dB\n', ...
                             i, i * update_rate*1000, m);
                end
            end
            
            if visualize_plots(7)
                % Vector with time samples
                time = linspace(0, last_snapshot * update_rate, ...
                                       last_snapshot);
                
                figure((start_idx+(k-1)*n_rx + f)*p);
                subplot(2,1,1);
                if manual_LOS && (length(segment_info(k).segment_index) > 1)
                    area_delimiter(time, pow, ...
                        segment_info(k).segment_index(2:end), length(time));
                    hold on;
                end
                plot(time,pow+0); % TX power is 0 dbm = 1mW
                title(['User = ', num2str(k), ', Mvmt param. = ', ...
                       num2str(rx_mvnt(k)),...
                       ', f = ', f_str, ' GHz, STDeviation = ', ...
                       num2str(std(pow))]);
                xlabel('Time [s]');
                ylabel('RX power (dBm)'); grid on;
                x_max = max(time);
                axis([0, x_max, min(pow)-5, max(pow)+5]);

                subplot(2,1,2);
                if manual_LOS && (length(segment_info(k).segment_index) > 1)
                    area_delimiter(time, pow_percent_in_LoS, ...
                        segment_info(k).segment_index(2:end), length(time));
                    hold on;
                end
                plot(time, pow_percent_in_LoS); % TX power is 0 dbm = 1mW
                title(['User = ', num2str(k), ', Mvmt param. = ', ...
                       num2str(rx_mvnt(k)), ...
                       ', f = ', f_str, ...
                       ' GHz, STDeviation of % of power in LoS= ', ...
                       num2str(std(pow_percent_in_LoS))]);
                xlabel('Time [s]');
                ylabel('Los/Total RSS'); grid on;
                xlim([0, x_max]);
                if min(pow_percent_in_LoS) == max(pow_percent_in_LoS)
                    if min(pow) == max(pow)
                        disp(['There are no changes in the signal.', ...
                          'Did you input a very small trim interval?']);
                    end
                    % if the signal changes, then the scenario is probably
                    % just LoS, hence the non variability of the pow %.
                else
                    ylim([min(pow_percent_in_LoS), max(pow_percent_in_LoS)]);
                end


            end
        end
    end
    end
    

end

