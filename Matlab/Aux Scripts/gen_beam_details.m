F = [ 3.5e9, 26e9 ];                % Frequency
N_v = [4, 8];                            % Number of vertical antenna elements
N_h = [4, 8];                            % Number of vertical antenna elements
N = N_v .* N_h;                      % #Total antenna elements
beam_details = zeros(2,121,6);        % Final precoded beam details 
                                    % 1.HPBW-AZ,
                                    % 2. HPBW-EL, 
                                    % 3. max direction (AZ), 
                                    % 4. max direction (EL)
                                    % 5. Linear Amplitude gain
                                    % 6. Logarithmic power gain
precoders_folder = ...
    'C:\Users\kizhakkekundils\Documents\THESIS\SXRSIMv3\SXRSIMv3\Matlab\Precoders';

for f = 1 : size(F, 2)              % frequency index for beam details.
    k = 1;                          % index for saving beam details.  
    for i = 1 : 11
        for j = 1 : 11
            % Loading precoder for 4 UEs.
            filename = [num2str(N_v(f)), '_', num2str(N_h(f)), ...
                        '_-60_60_12_0_-60_60_12_0_pol_1.mat'];
            precoders_array = ...
                load([precoders_folder, '\precoders_', filename]).precoders_array;
            a(f) = qd_arrayant('3gpp-3d', N_v(f), N_h(f), F(f), 1);
            a(f).coupling = ones(N(f), 1); % or other precoder (like MRT)
            p = squeeze(precoders_array(i,j,:)); 
            a(f).coupling = p; %#ok<*SAGROW>
            a(f).combine_pattern();

            [ C, I ] = max(abs(a(f).Fa));
            [ ~, el_idx_of_max ] = max(C);
            azi_idx_of_max = I(el_idx_of_max);  %cross check once.

            beam_details(f, k, 5) = abs(a(f).Fa(azi_idx_of_max,el_idx_of_max));
            beam_details(f, k, 6) = mag2db(beam_details(f, k, 5));

            disp(['Gains for a ', num2str(N_v(f)), ' x ', ...
                                  num2str(N_h(f)), ' array.']);
            disp(['Linear Amplitude gain: ', num2str(beam_details(f, k, 5))]);
            disp(['Logarithmic Power gain: ', ...
                              num2str(mag2db(beam_details(f, k, 5))), ' dB']);
            disp(newline)

            [ beam_details(f, k,1), beam_details(f, k,2), az_max_ang, el_max_ang ]...
                = a(f).calc_beamwidth();

            beam_details(f, k, 3) = round(az_max_ang, 4);
            beam_details(f, k, 4) = round(el_max_ang, 4);

            disp(['HPBW in azimuth: ', num2str(beam_details(f, k, 1))]);
            disp(['HPWB in elevation: ', num2str(beam_details(f, k, 2))]);
            disp(['Maximum direction: ', num2str([round(az_max_ang, 4), ...
                                                  round(el_max_ang, 4)])]);

            disp(newline)
            k = k + 1;
        end
    end
    save(['beam_details_', filename], 'beam_details');
end


