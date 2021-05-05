% SIMULATED PRECODERS

diff_pol = true;
save_p = true;
plot_p = false;
do_vids = false;

% Example:
gen_grid_of_beams([4 4],    ... % arr_size
                  [-60 60], ... % azi_lims
                  [-60 60], ... % el_lims
                  30,       ... % azi_res
                  30,       ... % el_res
                  0,        ... % azi_offset
                  0,        ... % el_offset
                  diff_pol, ... % diff_polarisations
                  0.5,      ... % elect_ele_space
                  plot_p,   ... % plot_precoders
                  save_p,   ... % save_precoders
                  do_vids,  ... % do_vid
                  0,        ... % use_taper
                  -20,      ... % sll
                  4);           % nbar

% IMPORTANT!
% Be careful choosing resolution, limits and offset values!
% Make it so that the limits, resolutions and offsets line up perfectly. 
% (limit - offset)/res = integer. 
% E.g. offset = 12, limit = 60, res = 24.-> points [12, 36, 60]
%      if the offset was 10, it would break before the offset in the 
%      ascending loop: [-60, -50, -40, -30, -20]
%%
% 3.5 GHz with 16 dual-polarised antenna elements
gen_grid_of_beams([4 4], 'auto1', [-60 60], [-60 60], 12, 12, 0, 0, save_p);
              %%
% 26 GHz with 64 dual-polarised antenna elements
gen_grid_of_beams([8 8], 'auto1', [-60 60], [-60 60], 12, 12, 0, 0, save_p);

% Comm
              
% With tapers:
% gen_grid_of_beams([8 8], [-60 60], [-60 60], 15, 15, diff_pol, 0.5, plot_p, save_p, do_vids, 1, -20, 4);
% gen_grid_of_beams([8 8], [-60 60], [-60 60], 15, 15, diff_pol, 0.5, plot_p, save_p, do_vids, 1, -20, 8);
% gen_grid_of_beams([8 8], [-60 60], [-60 60], 15, 15, diff_pol, 0.5, plot_p, save_p, do_vids, 1, -30, 4);

%%