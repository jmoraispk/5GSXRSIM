% One 3GPP element with 1 W Tx power has % 2.5119 linear (amplitude) gain. 
% The total (linear) gain for 32 elements is 80.3804 (2.5119 * 32 = 80.3804)

% Power gains = 10 * log10(a^2) = 20 * log10(a)[dB]
%             = 8dB and 38.1030 dB, respectively for 1 element and 32, 
%               assuming the 32-element array uses the same total power as
%               the 1 3GPP element alone.

% 32 is a superposition in the linear domain, so the power gain will
% be square: mag2db(32) = 20 * log10(32) = 30.1030

% Therefore 8dB + 30.103 = 38.103 = gain of 32 element array

% Note: the 3GPP element is dual-polarised. 

N_v = 4;
N_h = 4;
N = N_v * N_h;

a = qd_arrayant('3gpp-3d', N_v, N_h, 3.5e9, 1);
a.coupling = ones(N, 1); % or other precoder (like MRT)
%a.coupling = p;
a.combine_pattern();
a.visualize();


% TODO: test the subarray function! a.sub_array(1:64);

[C,I] = max(abs(a.Fa));
[~,el_idx_of_max] = max(C);
azi_idx_of_max = I(azi_idx_of_max);

lin_gain = abs(a.Fa(azi_idx_of_max,el_idx_of_max));

disp(['Gains for a ', num2str(N_v), ' x ', num2str(N_h), ' array.']);
disp(['Linear Amplitude gain: ', num2str(lin_gain)]);
disp(['Logarithmic Power gain: ', num2str(mag2db(lin_gain)), ' dB']);
disp(newline)

[ beamwidth_az, beamwidth_el, az_max_ang, el_max_ang ] =...
    a.calc_beamwidth();

disp(['HPBW in azimuth: ', num2str(beamwidth_az)]);
disp(['HPWB in elevation: ', num2str(beamwidth_el)]);
disp(['Maximum direction: ', num2str([round(az_max_ang,4), ...
                                      round(el_max_ang,4)])]);

disp(newline)


%% Sandra's Cell (rename :p)





