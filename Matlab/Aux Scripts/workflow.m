%% Parallel Simulation: Setup Phase

n_div = 4;
n_freq = 2; n_rx = 1; n_tx = 8;
parallelisation_level = 3;
if parallelisation_level == 0
    n_instances_per_time_division = 1;
elseif parallelisation_level == 1
    n_instances_per_time_division = n_freq;
elseif parallelisation_level == 2
    n_instances_per_time_division = n_freq * n_tx;
elseif parallelisation_level == 3
    n_instances_per_time_division = n_freq * n_tx * n_rx;
else
    error('Only parallelisation levels from 0 to 3 are available');
end

% Setup + save
% Meeting11('MatlabInput.mat', 2, [2,n_div]);
Meeting12('MatlabInput.mat', 2, [parallelisation_level, n_div]);
%% Parallel Simulation: Channel computation, Blockage and Aggregation

disp('Begin Simulation at: ');
disp(datetime('now'));

% Channel Computation
s = 'Sim_2021-03-08_11h01m52s_SEED3\';
n_inst = (n_instances_per_time_division * n_div);
%n_inst = 1;
for i = 1: 1
%     Meeting11(s, 3, [parallelisation_level, i]);
    Meeting12(s, 3, [parallelisation_level, i]);
    % For blockage computation in parallel.
    %Meeting11(s, 4, [parallelisation_level, i]);
end

% For blockage computation in series use 0!
%Meeting11(s, 4, [parallelisation_level, 0]);
%Meeting11(s, 5, '');


disp('Simulation Finished at: ');
disp(datetime('now'));

%% Series Simulation

%Meeting11('MatlabInput.mat', 0, '');
Meeting12('', 0, '');

%% Other Stuff
%% This is how to read one numerology only, YOU HAVE TO KNOW THE DIMENSIONS
n = 1; 
fr_full = read_complex([mother_folder, 'fr_blocked_full_num_', num2str(numerology(n))], ...
                       [base_channel_dimensions, 8, 18, 36, 16004]);


%% Code to apply blocking sequentially (Sandra is only using 1 Matlab instance)

% The first place in instance info doesn't matter
% but the flow_control must be 4, and the folder must contain the channels
% in the format they are generated.
Meeting11('folder with channels', 4, [231, 0]); 

%% NOTE FOR COMPARING BETWEEN SIMULATIONS:
% Open the vars.mat in matlab's comparison tool: visdiff(.)
% It shows the differences between each of the simulations

% NOTE2: if there are differences between simulations is because the
% variables are different. Otherwise there are no reasons for differences.
% This has been tested: Same var file, same output. Always.