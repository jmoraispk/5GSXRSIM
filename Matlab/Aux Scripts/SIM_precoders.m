% SIMULATED PRECODERS

diff_pol = true;
save_p = true;
plot_p = false;
do_vids = false;

%%
% Example (DO NOT USE, SIMPLY TO EXPLAIN THE ARGUMENTS):
gen_grid_of_beams([4 4],    ... % arr_size
                  'auto1',  ... % mode of deriving beamforming angles
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

%% For Cross-layer optimization (1 omnidirectional element only)

% We save the directions instead of the azi and el values
% 2 x N BEAMS matrix.
precoders_directions = [0;0];
precoders_matrix = [1];
n_azi_beams = 1;
n_ele_beams = 1;

save('1-omni-element', 'precoders_matrix', 'precoders_directions',...
                       'n_azi_beams', 'n_ele_beams');


%% With Tapers
sll_vals = [-20, -30, -40];
nbar_vals = 1:10; % Note that there are only 4 or so side lobes...
                      
% With tapers enabled
for sll = sll_vals
    for nbar = nbar_vals
        gen_grid_of_beams([8 8], [-30 -30], [-30 -30], 15, 15, false, 0.5, ...
                          true, false, false, 1, sll, nbar);
    end
end

% Conclusions: 
% - nbar should be at > 1, 1 is seems to be the main lobe index;
% - nbar does nothing when it becomes bigger than the number of sidelobes;
% - within decent range, a higher nbar leads to a narrower mainlobe, at 
%   the cost of more side lobes being increased to SLL (normally they 
%   would be smaller);
% - sll tells how lower the biggest side lobe should be, and has a direct
%   impact on the width of the main lobe;

% For the grid of beams:
%    - pick nbar = max(array_size)/2 -> this way is big enough for sure.
%    - pick a few side lobe levels
