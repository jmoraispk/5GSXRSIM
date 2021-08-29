Fin_w1 = precoders_matrix(:,175)+ (sqrt(0.5).*  precoders_matrix(:,179))+...
    (sqrt(0.25).*  precoders_matrix(:,171))+(sqrt(0.125).*  precoders_matrix(:,167));
Fin_w2 = precoders_matrix(:,43)+ (sqrt(0.5).*  precoders_matrix(:,235))+...
    (sqrt(0.25).*  precoders_matrix(:,107))+(sqrt(0.125).*  precoders_matrix(:,171));

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
qd_3gpp_arr_linear = qd_arrayant('3gpp-3d',...
                              N2,...             % 1 element in vertical
                              N1,...             % 1 element in horizontal
                              fc,...            % freq centre
                              pol_indicator,... % polarisation
                              0,...             % electrical down-tilt
                              elect_ele_space(1)); 
                          
                          
 qd_3gpp_arr_linear.coupling = Fin_w1;
 qd_3gpp_arr_linear.combine_pattern;
 
 [ beam_details_1, beam_details_2, ...
          az_point_ang1, el_point_ang1 ] = ...
                               calc_beamwidth( qd_3gpp_arr_linear, 1, 3 );