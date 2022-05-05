% Creates the array where each colm contains precoders belonging to each
% rot factor (16 rot factors (16 colm) each containing 16 precoders (16
% rows)
N1 = 8;
N2 = 8;
O2 = 4;
O1 = 4;
gob_col_size = N2 * O2;
q_idxs = zeros(64,16);

for a = 0 : 15
    q = a;
    if (4 <= q) && (q <= 7)
        q = q + (gob_col_size - O2) * 1;
    else
        if (8 <= q) && (q <= 11)
                q = q + (gob_col_size - O2) * 2;
        else
            if (12 <= q) && (q <= 15)
                    q = q + (gob_col_size - O2) * 3;
            end
        end
    end

    %   Step 2: Sum 'offsets' to get the remaining beams in the set
        q_col_idxs = [q + ([0:N1-1] .* (N2*O2*O1))];
        i = 1;
        q_idxs_list = zeros(size(q_col_idxs, 2), 8);
        for q_idx = q_col_idxs
            q_idxs_list(i, 1:end) = [q_idx + [0:N1-1] .* 4];
            i = i+1;
        end
        q_idxs(:,a+1) = sort(q_idxs_list(:));
end
% Final rot factor precoders
% q_idxs = q_idxs + 1 ;
%% Plot pointing direction of each rot factor beams.   

load('precoders_4_4_4_4_pol_3_RI_1_ph_1.mat', 'precoders_directions');
% c = [0 1 0; 1 0 0; 0.5 0.5 0.5; 0.6 0 1];
% clrs = parula(16);
% colororder(clrs)
% c = c+0.2;

   for x = 1 : 16
%        figure(x)
%        subplot(4, 4, x)
       xaxis = -180:180;
       yaxis = -90:90;
       for j = q_idxs(:, x)
           scatter(precoders_directions(1, j), precoders_directions(2, j), 10*x, 'filled');
       end
%        xlabel('Azimuth angle (\circ)')
%        ylabel('Elevation angle (\circ)')
%        xlim([-60, 60])
%        ylim([-60, 60])
%        gravstr = sprintf('R%i', round(x));
%        legend(gravstr)
       hold on
   end
   xlabel('Azimuth angle (in deg)')
   ylabel('Elevation angle (in deg)')
   legend('R1', 'R2','R3', 'R4','R5', 'R6', 'R7','R8', 'R9', ...
            'R10', 'R11', 'R12', 'R13', 'R14','R15', 'R16')
%% Checking orthogonality of different beams among single rot factor.
precod_dot = zeros(16, 16, 16);

for factor = 1 : 16
    rot = q_idxs(:, factor);
    
    for p1 = 1 : length(rot) 

        for p2 = 1 : length(rot)

            norm_precod_1 = precoders_matrix(:, rot(p1)) ./ norm(precoders_matrix(:, rot(p1)));
            norm_precod_2 = precoders_matrix(:, rot(p2)) ./ norm(precoders_matrix(:, rot(p2)));
        %     axres(p2+1) = abs(transpose(ax1)*ax2);
            precod_dot(factor, p1, p2) = dot(norm_precod_1, norm_precod_2);
            if precod_dot(factor, p1, p2) > 0.1 && rot(p1)~= rot(p2)
                fprintf('rot(p1) = %i, rot(p2) = %i\n',rot(p1), rot(p2))
                disp(precod_dot(factor, p1, p2) );  
            end

        end
    end
    precod_dot(precod_dot < 0.001) = 0;
end
%% orthogonality check for LC = 2

% Selected precoders
precod_col_P1_L1 = precoders_matrix(:,111);
precod_col_P1_L2 = precoders_matrix(:,99);

% precod_col_P2_L1 = precoders_matrix(:,251);
precod_col_P2_L1 = precoders_matrix(:,43);
precod_col_P2_L2 = precoders_matrix(:,235);
% precod_col_P2_L2 = precoders_matrix(:,63);

% RPI
% precod_amp2_P1 = 0.3716379337316983;
% precod_amp2_P2 = 0.5326218226414174;

precod_amp2_P1 = 0.25;
precod_amp2_P2 = 0.5;
% Precoder L = 2
P1_L2 = (1/(sqrt(1+(precod_amp2_P1 * precod_amp2_P1)))).* ...
            (precod_col_P1_L1 + (precod_amp2_P1.* precod_col_P1_L2));    
norm_P1_L2 = P1_L2 ./ norm(P1_L2);

P2_L2 = (1/(sqrt(1+(precod_amp2_P2 * precod_amp2_P2)))).* ...
            (precod_col_P2_L1 + (precod_amp2_P2.* precod_col_P2_L2));    
norm_P2_L2 = P2_L2 ./ norm(P2_L2);

dot_product_P1_P2 = abs(dot(norm_P1_L2, norm_P2_L2));

precod_applied = norm_P1_L2;
elect_ele_space = [1/2, 1/2];
qd_3gpp_arr_linear = qd_arrayant('3gpp-3d',...
                              4,...             % 1 element in vertical
                              4,...             % 1 element in horizontal
                              1e9,...            % freq centre
                              3,... % polarisation
                              0,...             % electrical down-tilt
                              elect_ele_space(1)); % element spacing, [lambda]
qd_3gpp_arr_linear.coupling = precod_applied;

qd_3gpp_arr_linear.combine_pattern;
[direc, fin_gain] = calc_gain(qd_3gpp_arr_linear);
[ HPBW_AZ_TX, HPBW_EL_TX, az_point_ang1, el_point_ang1 ] = ...
                               calc_beamwidth( qd_3gpp_arr_linear, 1, 3 );
                           %% Polar plot, pointing angles etc.

az_ang_deg_range = -180 : 180;
el_ang_deg_range = -90 : 90;
% P = abs(qd_3gpp_arr_linear.Fa).^ 2 + abs(qd_3gpp_arr_linear.Fb).^ 2;
% P_max = max( P(:) );

az_point_ang1 = round(az_point_ang1, 1);
el_point_ang1 = round(el_point_ang1, 1);
az_max_gain_ang_indx = find(az_ang_deg_range == az_point_ang1);
el_max_gain_ang_indx = find(el_ang_deg_range == el_point_ang1);
P1_el = abs(qd_3gpp_arr_linear.Fa(:, az_max_gain_ang_indx)).^ 2 ...
                + abs(qd_3gpp_arr_linear.Fb(:, az_max_gain_ang_indx)).^ 2;

P1_el_db = 10 * log10(P1_el);
P1_el_db(P1_el_db < 0) = 0;
polarplot(el_ang_deg_range * (pi / 180), P1_el_db);
hold on