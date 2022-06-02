% figure()
% Fin_w1_L2 = (1/(sqrt(1+(1 * 1)))).* ...
%         (precoders_matrix(:,11)+ (1.*  precoders_matrix(:,12)));
% Fin_w1_L1 = precoders_matrix(:,11);
% Fin_w1_L3 = (1/(sqrt(1+(1 * 1)+(0.7071067811 * 0.7071067811)))).* ...
%         (precoders_matrix(:,11)+ (1.*  precoders_matrix(:,12))+ ...
%             0.7071067811 .* precoders_matrix(:,10));
% Fin_w1_L4 = (1/(sqrt(1+(1 * 1)+(0.7071067811 * 0.7071067811)+(0.7071067811 * 0.7071067811)))).* ...
%     (precoders_matrix(:,11)+ (1.*  precoders_matrix(:,12))+ ...
%             (0.7071067811 .* precoders_matrix(:,10))+ (0.7071067811 .* precoders_matrix(:,13)));


Fin_w1_L2 = (1/(sqrt(1+(0.5 * 0.5)))).* ...
        (precoders_matrix(:,43)+ (0.5.*  precoders_matrix(:,235)));
    
norm_L2 = ((1/sum(abs(Fin_w1_L2))).*Fin_w1_L2).*4;

Fin_w1_L1 = precoders_matrix(:,43);
norm_L1 = ((1/sum(abs(Fin_w1_L1))).*Fin_w1_L1).* 4;

Fin_w1_L3 = (1/(sqrt(1+(0.5 * 0.5)+(0.125 * 0.125)))).* ...
        (precoders_matrix(:,43)+ (0.5.*  precoders_matrix(:,235))+ ...
            (0.125 .* precoders_matrix(:,107)));
norm_L3 = ((1/sum(abs(Fin_w1_L3))).*Fin_w1_L3).*4;

Fin_w1_L4 = (1/(sqrt(1+(0.5 * 0.5)+(0.125 * 0.125)+(0.125 * 0.125)))).* ...
    (precoders_matrix(:,43)+ (0.5.*  precoders_matrix(:,235))+ ...
            (0.125 .* precoders_matrix(:,107))+ (0.125 .* precoders_matrix(:,171)));
norm_L4 = ((1/sum(abs(Fin_w1_L4))).*Fin_w1_L4).*4;

Fin_w1_L5 = (1/(sqrt(1+(0.5 * 0.5)+(0.75 * 0.5)))).* ...
        (precoders_matrix(:,43)+ (0.5.*  precoders_matrix(:,235))+ ...
            (0.5 .* precoders_matrix(:,107)));
norm_L5 = ((1/sum(abs(Fin_w1_L3))).*Fin_w1_L5).*4;
qd_3gpp_arr_linear = qd_arrayant('3gpp-3d',...
                              4,...             % 1 element in vertical
                              4,...             % 1 element in horizontal
                              fc,...            % freq centre
                              pol_indicator,... % polarisation
                              0,...             % electrical down-tilt
                              elect_ele_space(1)); % element spacing, [lambda]
qd_3gpp_arr_linear.coupling = norm_L3;

qd_3gpp_arr_linear.combine_pattern;
qd_3gpp_arr_linear.visualize;
%%
az_ang_deg_range = -180 : 180;
el_ang_deg_range = -90 : 90;
P = abs(qd_3gpp_arr_linear.Fa).^ 2 + abs(qd_3gpp_arr_linear.Fb).^ 2;
        P_max = max( P(:) );

[ HPBW_AZ_TX, HPBW_EL_TX, az_point_ang1, el_point_ang1 ] = ...
                               calc_beamwidth( qd_3gpp_arr_linear, 1, 3 );
az_point_ang1 = round(az_point_ang1, 2);
el_point_ang1 = round(el_point_ang1, 2);
az_max_gain_ang_indx = find(az_ang_deg_range == az_point_ang1);
el_max_gain_ang_indx = find(el_ang_deg_range == el_point_ang1);
P1_el = abs(qd_3gpp_arr_linear.Fa(:, az_max_gain_ang_indx)).^ 2 ...
                + abs(qd_3gpp_arr_linear.Fb(:, az_max_gain_ang_indx)).^ 2;

P1_el_db = 10 * log10(P1_el);
P1_el_db(P1_el_db < 0) = 0;
polarplot(el_ang_deg_range * (pi / 180), P1_el_db);
% hold on
% figure()
P1_az = abs(qd_3gpp_arr_linear.Fa(el_max_gain_ang_indx, :)) .^ 2 + ...
                   abs(qd_3gpp_arr_linear.Fb(el_max_gain_ang_indx, :)) .^ 2;

P1_az_db = 10 * log10(P1_az);
P1_az_db(P1_az_db < 0) = 0;
polarplot(az_ang_deg_range * (pi / 180), P1_az_db);
hold on