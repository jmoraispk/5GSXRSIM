N1 = 4;     % Number of logical antenna elements in hor direction.            
N2 = 4;     % Number of logical antenna elements in ver direction. 
O1 = 4;     % Oversampling factor in horizontal direction: O1 is always 4.
O2 = 4;     % Oversampling factor in vertical direction: O2 can be 1 or 4.

pol_indicator = 3;          % dual-pol ant array (2 coefts per cross ele).
RI = 1;                           % Number of transmission layers.

fc = 1e9;                         % Center frequency.
elect_ele_space = [1/2, 1/2];     % Inter element spacing: ant array.
cophase_fact = [1, -1, 1i, -1i];  % Cophasing factor used for dual pol(3).
cophase_fact = 1;

save_precoders = 0;        % If you need to save precoders with max angles.
Need_plot = 1;             % polar plots (az & el cuts)
el_cut = 0;                % 1: el cut polar plot, 0: az cut.
Quad_3d_plot = 1;
debug_gain = 1;
% precoder = 1:256;
precoder = 1;
%%
% precoder = [1, 5, 9, 13, 65, 69, 73, 77];
% precoder = [1, 65, 65+64, 65+64+64];
% a=16;
% precoder = [1+a, 5+a, 9+a, 13+a];
% precoder = 1 : 1 : 256;
% precoder = [9, 25, 41, 57, 73];
% precoder = [1, 5, 9, 13];
% precoder = [2, 6, 10, 14];
% precoder = [3, 7, 11, 15];
% precoder = [4, 8, 12, 16];

% figure(2)

az_point_ang1 = zeros(length(cophase_fact), length(precoder));
el_point_ang1 = zeros(length(cophase_fact), length(precoder));
directivity_dBi = zeros(length(cophase_fact), length(precoder));
gain_dBi = zeros(length(cophase_fact), length(precoder));
beam_details = zeros( length(fc), N1 * O1 * N2 * O2, 6);% Precoded beam details 
                                              % 1. HPBW-AZ,
                                              % 2. HPBW-EL, 
                                              % 3. max direction (AZ), 
                                              % 4. max direction (EL)
                                              % 5. Linear Amplitude gain
                                              % 6. Logarithmic power gain

for p = 1 : length(cophase_fact)
    % Fetch the DFT codebook according to the above set param.
    W = DFT_codebook( N1, N2, O1, O2, pol_indicator, RI, cophase_fact(p) );
    % Re-order the above codebook to match the antenna arrangement in
    % Quadriga. Verified using element positions of Quadriga. Order is 1st
    % element of 1st pol, 1st element of 2nd pol, 2nd element of 1st pol,
    % 2nd element of 2nd pol etc..
    W2 = zeros( size(W) );
    if pol_indicator == 3
        oddI = 1 : 2 : size(W, 1);
        evenI = 2 : 2 : size(W, 1);
        x1 = W(1 : (N1 * N2), :);
        x2 = W( (N1 * N2) + 1 : end, :);
%         W2(oddI, :) = double_flip_matr(x1);
%         W2(evenI,:) = double_flip_matr(x2);
        W2(oddI, :) = x1;
        W2(evenI,:) = x2;
    else
        W2 = 1 .* W;
        disp('hello')
    end  
    
    if Need_plot
        subplot(2, 2, p);
    end
    
    for i = 1  : 256
%     for i = 1 : N1 * O1 * N2 * O2
        
        % Creating quadriga antenna array (ULA/URA with single/dual pol).
        qd_3gpp_arr_linear = qd_arrayant('3gpp-3d',...
                              N2,...             % 1 element in vertical
                              N1,...             % 1 element in horizontal
                              fc,...            % freq centre
                              pol_indicator,... % polarisation
                              0,...             % electrical down-tilt
                              elect_ele_space(1)); % element spacing, [lambda]
        
        % Apply precoder from DFT codebook, ith col of precoder applied.
        % Quadriga indexes the elements the opposite way of Phased Array
        % (and the opposite way of 3GPP), that's why we flip it before.
        qd_3gpp_arr_linear.coupling = W2(:, i);
        qd_3gpp_arr_linear.combine_pattern;
        
        if Need_plot && Quad_3d_plot
            qd_3gpp_arr_linear.visualize();
        end
        
        % az & el angle range.
        az_ang_deg_range = -180 : 180;
        el_ang_deg_range = -90 : 90;
                
        % Ignore HPBW, for obtaining the az&el angles (in deg) in direction
        % of the max gain.
        [ beam_details(1, i, 1), beam_details(1, i, 2), ...
          az_point_ang1(p, i), el_point_ang1(p, i) ] = ...
                               calc_beamwidth( qd_3gpp_arr_linear, 1, 3 );
                
        % Display az and el angles in degrees for max gain direction.
        az_point_ang1(p, i) = round(az_point_ang1(p, i), 1);
        el_point_ang1(p, i) = round(el_point_ang1(p, i), 1);
        
        beam_details(1, i, 3) = az_point_ang1(p, i);
        beam_details(1, i, 4) = el_point_ang1(p, i);
%         disp( az_point_ang1 );
%         disp( el_point_ang1 );

            % Find index of max beam pointing az & el angles in the ang range.
            az_max_gain_ang_indx = find(az_ang_deg_range == az_point_ang1(p, i));
            el_max_gain_ang_indx = find(el_ang_deg_range == el_point_ang1(p, i));
            
        if debug_gain
            % Find max gain of the combined antenna element.   
            P = abs(qd_3gpp_arr_linear.Fa).^ 2 + ...
                abs(qd_3gpp_arr_linear.Fb).^ 2;
            P_max = max( P(:) );
            
            % This is just to verify that the max gain for the obatined az and
            % el angles is same as the one obatined in P_max above.
            P_max_compare = P(el_max_gain_ang_indx, az_max_gain_ang_indx);
            [directivity_dBi(p, i), gain_dBi(p, i)] = calc_gain(qd_3gpp_arr_linear);
            beam_details(1, i, 5) = 0.1 * 10 ^ (gain_dBi(p, i));
            beam_details(1, i, 6) = gain_dBi(p, i);
        end
        
        
        disp(i)
        if Need_plot
            if el_cut
                % Elevation cut plot per precoder beamforming. 
               P1_el = abs(qd_3gpp_arr_linear.Fa(:, az_max_gain_ang_indx)).^ 2 ...
                     + abs(qd_3gpp_arr_linear.Fb(:, az_max_gain_ang_indx)).^ 2;

               P1_el_db = 10 * log10(P1_el);
               P1_el_db(P1_el_db < 0) = 0;
               polarplot(el_ang_deg_range * (pi / 180), P1_el_db);
               hold on
            else

               % Azimuth cut plot per precoder beamforming. 
               P1_az = ...
                   abs(qd_3gpp_arr_linear.Fa(el_max_gain_ang_indx, :)) .^ 2 ...
                   + abs(qd_3gpp_arr_linear.Fb(el_max_gain_ang_indx, :)) .^ 2;

                P1_az_db = 10 * log10(P1_az);
                P1_az_db(P1_az_db < 0) = 0;
                polarplot(az_ang_deg_range * (pi / 180), P1_az_db);
        %         f = gcf();
        %         f.Position = [0.0500 0.2650 1.2000 0.5176] * 1000;
                hold on
            end
        end

    end
    if Need_plot
        title(["cophase factor", num2str( cophase_fact(p) )]);
    end
    gob_name = [num2str(N1), '_', ...
                num2str(N2), '_', ...
                num2str(O1), '_', ...
                num2str(O2), '_', ...
                'pol_', num2str(pol_indicator), '_', ...
                'RI_', num2str(RI), '_', ...
                'ph_', num2str(cophase_fact(p))];

    if save_precoders
            n_azi_beams = N1 * O1;
            n_ele_beams = N2 * O2;
            precoders_directions = [flip(az_point_ang1(p, :)); 
                                    el_point_ang1(p, :)];
            precoders_matrix = W2;
            
            disp('Saving in a file...');
            save_file_name = ['precoders_', gob_name];
            save(save_file_name, 'precoders_matrix', ...
                                 'precoders_directions', 'n_azi_beams',...
                                                         'n_ele_beams');
            beam_details = squeeze(beam_details);         
            save(['beam_details_', gob_name], 'beam_details');                                         
    end
end

if Need_plot && el_cut
    sgtitle('Elevation cut');
elseif Need_plot && ~el_cut
    sgtitle('Azimuth cut');
end

% legend("W = 1","W = 2","W = 3","W = 4","W = 5" )
% legend("W = 1","W = 5","W = 9","W = 13")%"W = 65", "W = 69", "W = 73", "W = 77")
% legend("W = 2","W = 6","W = 10","W = 14")
% % legend("W = 3","W = 7","W = 11","W = 15")
% legend("W = 9","W = 25","W = 41","W = 57","W = 73")
