function [az_cut_pat, az_vals] = ArrayTester_v2(varargin)
% ArrayTester_v2([30;0])

%% The Array Tester

%[# V elements, # H elements, # V elements in subarray , # H elements in subarray]
arrays = {[2,2,1,1], [4,4,1,1], [8,8,1,1], [16,16,1,1], ...
          [12,12,2,1], [12,12,3,1], [12,12,4,1], [12,12,6,1], [12,12,12,1],...
          [2,4,2,1], [4,4,2,1], [6,4,2,1], [8,4,2,1], ...
          [4,8,2,1], [4,16,2,1], ...
          [8,4,4,1], [8,8,4,1], [8,16,4,1], ...
          [16,4,4,1], [16,8,4,1], [16,16,4,1],...
          [16,4,8,1], [16,8,8,1], [16,16,8,1],...
          [4,4,2,2], [8,8,2,2], [8,8,4,4], [16,16,2,2], [16,16,4,4], [16,16,8,8], ...
          [4,2,1,1], [8,2,1,1], [16,2,1,1], [8,4,1,1], [16,4,1,1], [16,8,1,1,],...
          [32,32,1,1], [32,32,2,2], [32,32,4,4], [32,32,8,8], [32,32,16,16], ...
          [32,32,8,8]};
 
      
arrays = {[16,16,1,1], [16,16,2,2], [16,16,4,4], [16,16,8,8]};

% subarray NORMAL and POSITION are only possible when Layout is Custom      

% Used to test the subarray patterns (ignore):
% arrays = {[2,1,1,1], [3,1,1,1], [4,1,1,1], [6,1,1,1], ...
%           [8,1,1,1], [12,1,1,1], [16,1,1,1], [32,1,1,1]...
%           [2,2,1,1], [3,3,1,1], [4,4,1,1], [6,6,1,1], ...
%           [8,8,1,1], [12,12,1,1], [16,16,1,1], [32,32,1,1]};

% We need to steer the subarrays manually, or else all subarrays will have
% the same beam-steering vector applied to them.
subarrays_steering = repmat({'None'}, [1 41]);
subarrays_steering{end+1} = 'Custom';

% square_panel_pattern(4, 30)
% square_panel_pattern_diag(4, 30)
% flower_pattern(4, 30)
% HAVE THE FOLLOWING IN MIND WHEN PICKING ANGLES!
    % From the front, an array has the indices like: 
    % [1 5  9 13
    %  2 6 10 14
    %  3 7 11 15  
    %  4 8 12 16]
    % From the back, the indices would be like this:
    % [13  9  5 1
    %  14 10  6 2
    %  15 11  7 3  
    %  16 12  8 4]
% the three functions above already take this into consideration
% but in case you want to do it manually, you should remember


% when we need to test many directions: the direction of each element
steering_directions = {repmat({[10 0]}, 1, 4 * 4), ...
                       subarray_square_panel_pattern(4, 0), ...
                       subarray_square_panel_pattern(4, 5), ...
                       subarray_square_panel_pattern(4, 10), ...
                       subarray_square_panel_pattern(4, 15), ...
                       subarray_square_panel_pattern(4, 20), ...
                       subarray_square_panel_pattern(4, 25), ...
                       subarray_square_panel_pattern(4, 30), ...
                       subarray_square_panel_pattern(4, 35), ...
                       subarray_square_panel_pattern(4, 40), ...
                       subarray_square_panel_pattern(4, 45), ...
                       subarray_square_panel_pattern(4, 50), ...
                       subarray_square_panel_pattern(4, 55), ...
                       subarray_square_panel_pattern(4, 60), ...
                       subarray_square_panel_pattern_diag(4, 5), ...
                       subarray_square_panel_pattern_diag(4, 10), ...
                       subarray_square_panel_pattern_diag(4, 15), ...
                       subarray_square_panel_pattern_diag(4, 20), ...
                       subarray_square_panel_pattern_diag(4, 25), ...
                       subarray_square_panel_pattern_diag(4, 30), ...
                       subarray_square_panel_pattern_diag(4, 35), ...
                       subarray_square_panel_pattern_diag(4, 40), ...
                       subarray_square_panel_pattern_diag(4, 45), ...
                       subarray_square_panel_pattern_diag(4, 50), ...
                       subarray_square_panel_pattern_diag(4, 55), ...
                       subarray_square_panel_pattern_diag(4, 60), ...
                       subarray_flower_pattern(4, 5), ...
                       subarray_flower_pattern(4, 10), ...
                       subarray_flower_pattern(4, 15), ...
                       subarray_flower_pattern(4, 20), ...
                       subarray_flower_pattern(4, 25), ...
                       subarray_flower_pattern(4, 30), ...
                       subarray_flower_pattern(4, 35), ...
                       subarray_flower_pattern(4, 40), ...
                       subarray_flower_pattern(4, 45), ...
                       subarray_flower_pattern(4, 50), ...
                       subarray_flower_pattern(4, 55), ...
                       subarray_flower_pattern(4, 60), ...
                      };

% testing options: 
% - 'array': select different array and subarray structures 
% - 'analog_steering': For the same array and subarray structure, 
%                      test several different analog precoders for each of
%                      the subarrays;
% - 'digital steering': NOT IMPLEMENTED! This should be done in other
%                       function actually;


testing = 'array';
%testing = 'analog steering';
%testing = 'digital steering';

% Note: it may be useful to change the first loop line a bit more below: 
% this one: for loop_idx = 1:3 %[37, 42]% 1:loop_lim

fc = 1e9; lambda = physconst('LightSpeed')/fc;
elect_ele_space = [1/2, 1/2];
ele_space =  elect_ele_space .* lambda;


if nargin == 1
    steer_angle = varargin{1};
else
    % PLEASE, BE CAREFUL WITH THE FUCKING SEMI-COLON! 
    % THE FUCKING NIGHTMARES!!
    steer_angle = [0;0];
end

% Only print stats and don't plot?
do_plots = true;
save_plots = false;
% The following 2 are incompatible.
multiple_figs = false;
only_one_fig = true;
%
plot_3d_pattern = true;
plot_subarray = false;
print_precision = 1; % [deg]
plot_precision = 1;  % [deg]

plot_type = 'directivity';
    
%%%%%%%%%%%%%% DON'T TUNE FROM HERE ONWARDS! %%%%%%%%%%%%%%%%
% if the 3gpp dual-polarised element is only vertical polarised and the 
% response is computed jointly or if we take the separate
% polarisations of each of the single-polarised element. 1 or 3, respectiv.
pol_indicator = 1;    % ATTENTION DO WE WANT 1 CROSS POLARISED ELEMENT OR 2 SINGLE POLARISED ONES?!
inc_ele_resp = 1;

qd_3gpp_ele = qd_arrayant('3gpp-3d',...
                      1,...             % 1 element in vertical
                      1,...             % 1 element in horizontal
                      fc,...            % freq centre
                      pol_indicator,... % polarisation
                      0,...             % electrical down-tilt
                      elect_ele_space); % element spacing, [lambda]

if pol_indicator == 1
    a_crossed_3gpp = phased.CustomAntennaElement(...
        'MagnitudePattern', 10 * log10(qd_3gpp_ele.Fa .^ 2)); %[dB]
    ant_element_3gpp = a_crossed_3gpp;
else
    % Copy the left and right elements
    a_left =  phased.CustomAntennaElement(...
        'MagnitudePattern', 10 * log10(squeeze(qd_3gpp_ele.Fa(:,:,1)).^2));

    % a_right =  phased.CustomAntennaElement(...
    %   'MagnitudePattern', 10 * log10(squeeze(qd_3gpp_ele.Fa(:,:,2)).^2));
    
    % Note: since the precoder will be the same, we will only show for one
    %       of the elements.
    ant_element_3gpp = a_left;
end

if do_plots && ~multiple_figs && ~only_one_fig
    f = figure;
end

if strcmp(testing, 'array')
    % Normal loop over the several arrays
    loop_lim = length(arrays);
elseif strcmp(testing, 'analog steering')
    % Loop over the steering directions, maintaining the same array
    loop_lim = length(steering_directions);
end
    

for loop_idx = 1:1 %loop_lim
    if strcmp(testing, 'array')
        % Normal loop over the several arrays
        i = loop_idx;
        d_idx = 1;
    elseif strcmp(testing, 'analog steering')
        % Loop over the steering directions, maintaining the same array
        i = 42;
        d_idx = loop_idx;
    end
    
    disp(['Test ', num2str(i), ':', array_to_str(arrays{i})]);
    if strcmp(testing, 'analog steering')
        disp('Steering array to: ');
        disp(cell_to_str(steering_directions{d_idx}));
    end
    
    arr_size = arrays{i}(1:2)./arrays{i}(3:4);
    sub_array = arrays{i}(3:4);
    
    % Check if the inputs have correct values:
    if any(mod(arr_size,1))
        error(['Test number %d has wrong antenna dimensions. ', ...
               'The number of AEs should be divisible by the ', ...
               'sub array sizes'], i);
    end
    
    
    if isequal(sub_array, [1,1])
        sArray = phased.ConformalArray('Element', ant_element_3gpp);
    else
        if sub_array(2) == 1
            sArray = phased.ULA('Element', ant_element_3gpp, ...
                                'NumElements', sub_array(1), ...
                                'ElementSpacing', ele_space(1), ...
                                'ArrayAxis', 'z');
        else
            sArray = phased.URA('Element', ant_element_3gpp, ...
                                 'Size', sub_array, ...
                                 'ArrayNormal', 'x', ...
                                 'ElementSpacing', ele_space);
        end
    end
    
    if do_plots && plot_subarray
        f = figure;
        f.WindowState = 'maximized';
        sgtitle(sprintf('Sub array of %dx%d with no precoder', ...
                sub_array(1), sub_array(2)));
        subplot(1,3,1);
        pattern(sArray,fc, -180:print_precision:180, 0, ...
                'Type',plot_type);
        subplot(1,3,2);
        pattern(sArray,fc, 0, -90:print_precision:90, ...
                'Type',plot_type);
        subplot(1,3,3);
        pattern(sArray,fc, -180:print_precision:180, -90:print_precision:90, ...
                'Type',plot_type); zoom(2);
    end
    
    
    sRSA = phased.ReplicatedSubarray('Subarray',sArray,...
             'Layout','Rectangular',...
             'GridSize',arr_size,...
             'GridSpacing',[sub_array(1) * ele_space(1), ...
                            sub_array(2) * ele_space(2)], ...
             'SubarraySteering', subarrays_steering{i});
         
    sRSA_sv_obj = phased.SteeringVector('SensorArray', sRSA, ...
                                   'IncludeElementResponse', inc_ele_resp,...
                                   'EnablePolarization', false);
    
    if isequal(subarrays_steering{i}, 'None')
        sRSA_sv = sRSA_sv_obj(fc, steer_angle);
    else
        % compute the weights to apply to each of the elements for
        % analog precoding
        d = steering_directions{d_idx};
        
        analog_precoding_mat = [];
        for subarray_idx = 1:length(d)
            p = gen_steering_vector_URA(arrays{i}(3:4), ...
                                        d{subarray_idx}(1), ...
                                        d{subarray_idx}(2), ...
                                     true, elect_ele_space, false);
            analog_precoding_mat = [analog_precoding_mat p]; %#ok<AGROW>
        end
        
        sRSA_sv = sRSA_sv_obj(fc, steer_angle, analog_precoding_mat);
    end
    sRSA_sv = sRSA_sv ./ norm(sRSA_sv);
    
    % The first 'pattern' call computes information,
    % the second plots. Stupid, I know. 
    [az_cut_pat, az_vals, ~] = pattern(sRSA,fc,...
                                       -180:print_precision:180, ...
                                       steer_angle(2), ...
                                       'Type',plot_type,'Weights',sRSA_sv);
                                                                  
    % IMPORTANT WARNING: an internal function was changed to make this work.
    % phased.internal.parsePatternInputs parses the array of weights as 
    % a list of cells of cells {{16x1}, ..., {16x1}}, where each 16x1 is a 
    % precoder for a subarray. 
    
    % I am not sure if that function should be changed or if the function
    % AbstractSubarray.m:
    % It's line 250 of phased.internal.parsePatternInputs that is causing
    % the problem or lines 1422-1431 of AbstractSubarray.m.
    % CHANGES: added the following above the problem in AbstractSubarray.m:
%     if iscell(plotArgs)
%         n_elements_per_subarray = length(plotArgs{28}{1});
%         n_subarrays = length(plotArgs{28});
%         weights_matrix = zeros(n_elements_per_subarray, ...
%                                n_subarrays);
% 
%         for i = 1:n_subarrays
%             weights_matrix(:, i) = plotArgs{28}{i};
%         end
%             plotArgs{28} = weights_matrix;
%     end
                                   
                                   
    % PRINT HPBW, FNBW, DIRECTIVITY AND SLL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    disp_pattern_cut_information(az_cut_pat, az_vals);
    % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if do_plots
        if multiple_figs
            f=figure;  %#ok<*UNRCH>
        elseif only_one_fig
            f = figure();
            f.Position = [2 81 1279 401];
            if plot_3d_pattern
                subplot(1,3,1);
            else
                subplot(1,2,1);
            end
        end
        pattern(sRSA,fc,-180:plot_precision:180,steer_angle(2), ...
                                 'Type',plot_type,'Weights',sRSA_sv);
        if save_plots && ~only_one_fig
            % export with 300 pix per inch, and cut borders
            export_fig(f, '-c' , '[120,320,120,320]', '-r', '300', ...
               sprintf('Subarray testing\\test_%d_azicut.png', i))
        end
    end
    
    
    [el_cut_pat, ~, el_vals] = pattern(sRSA,fc,steer_angle(1), ...
                                       -90:print_precision:90, ...
                                       'Type',plot_type,'Weights',sRSA_sv);
    disp_pattern_cut_information(el_cut_pat, el_vals);
    
    
    % final pat has the 3d radiation pattern response
    % To get the gain/directivity -> check if gain or directivity....
    % you should do:
    % 1- find the index of elevation angle you want. el_vals has the angles
    % but you need the index of where the angle is in that vector!
    % such that el_vals(el_idx) = angle you want
    % 2- do the same for azimuth
    % 3- index the final_pat as final_pat(el_idx, azi_idx)
    
    [final_pat, azi_vals, el_vals] = pattern(sRSA,fc, ...
                                            -180:print_precision:180, ...
                                            -90:print_precision:90, ...
                                            'Type',plot_type,...
                                            'Weights',sRSA_sv);
                                        
	[max_at_each_el, el_idx_of_max] = max(final_pat, [], 1);
    [max_directivity, azi_idx_of_max] = max(max_at_each_el);
    
    fprintf(['Max directivity: %2.2f dBi\n', ...
             'Direction of max: [%2.2f, %2.2f]\n'], ...
             max_directivity, azi_vals(azi_idx_of_max), ...
             el_vals(el_idx_of_max(azi_idx_of_max)));
                                        
    if do_plots
        if multiple_figs
            f = figure; 
        elseif only_one_fig
            if plot_3d_pattern
                subplot(1,3,2);
            else
                subplot(1,2,2);
            end
        end
        pattern(sRSA,fc,steer_angle(1), -90:plot_precision:90, ...
                                 'Type',plot_type,'Weights',sRSA_sv);
        if ~only_one_fig
            export_fig(f, '-c' , '[120,320,120,320]', '-r', '300', ...
                   sprintf('Subarray testing\\test_%d_elcut.png', i));
        end
        if plot_3d_pattern 
            if multiple_figs
                f=figure; 
            elseif only_one_fig
                subplot(1,3,3);
            end
            pattern(sRSA,fc, -180:plot_precision:180, -90:plot_precision:90, ...
                    'Type',plot_type,'Weights',sRSA_sv); zoom(1.5);
            sgtitle(sprintf(['Array [%d,%d] steered to [%d, %d] º', ...
                             ' with subarray [%d, %d]'], ...
                             arrays{i}(1), arrays{i}(2), ...
                             steer_angle(1), steer_angle(2), ...
                             arrays{i}(3), arrays{i}(4)));
            if save_plots
                drawnow;
                if strcmp(testing, 'array')
                    s = sprintf('Subarray testing\\test_%d_3d.png', i);
                elseif strcmp(testing, 'analog steering')
                    s = sprintf(['Subarray testing\\subarray_steering_', ...
                                 'test_%d.png'], loop_idx);
                end
                exportgraphics(f, s, 'Resolution', 300);
            end
        end
    end
end

% Some matlab gold about warnings:
% Do:
%[msg, id] = lastwarn
%warning('off', id)
% To disable that warning from appearing in the console!!!


% MAGNITUDE TICKS is the name of the object in the axis: change limits!
end