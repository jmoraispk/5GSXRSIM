function [precoders_array] = gen_grid_of_beams(arr_size, ...
                                manual_or_auto, ...
                                azi_lims, el_lims, azi_res, el_res, ...
                                azi_offset, el_offset, save_precoders, ...
                                varargin)

    % Manual or auto = 
    %    - 'auto1':
    %    - 'auto2': 
    %    - 'manual': the azi_values and el_values are provided and every
    %                combination between them is computed.
                            
                            
	% IMPORTANT TODO: EXTEND THE PRECODER SUPPORT TO URAS AND TO SUBARRAYS!
    
    %   - with ULAS: simply check if the second dimension is 1, keep all the
    %                rest exactly the same.
    %   - with SUBARRAYS: add to the name, at the end, before the taper!
    %   NOTE: subarrays should ONLY matter because the distance between
    %         elements is different. In any case, for ploting, they need to
    %         be created. Check the ArrayTester function to copy how they
    %         are done. Note also that this function will generate only the
    %         digital precoding part of the hybrid array. If the subarrays
    %         are supposed to be steered to a certain direction, it gets
    %         more complicated (i.e. needing to receive each of those
    %         directions, which then happened in the array tester). 
    
    % NOTE FOR FORWARD COMPATIBILITY: 
    %    The precoder/GoB files need to be one per frequency and written
    %    in the IO section of the simulation parameters.
    
    % Generates the steering vectors (which will be used as precoders)
    % for a grid of beams given:
    % a) intervals and resolutions in azimuth and elevation
    %    (instead of resolutions, one can pick the 'number of beams'
    
    % enable this to see the progress of very extensive beam computations
    debug = 0;
    
    if nargin <=9
        diff_polarisations = 1;
        elect_ele_space = 0.5;
        plot_precoders = false;
        do_vid = false;
        use_taper = false;
        sll = 0;
        nbar = 0;
        angles = {};
    else
        diff_polarisations = varargin{1}; 
        elect_ele_space = varargin{2};
        plot_precoders = varargin{3};
        do_vid =  varargin{4};
        use_taper =  varargin{5};
        sll = varargin{6};
        nbar = varargin{7};
        angles = varargin{8};
    end
    
    if strcmp(manual_or_auto, 'auto1') || strcmp(manual_or_auto, 'auto2')
        % Set some variables
        azi_interval = azi_lims(2) - azi_lims(1);
        ele_interval = el_lims(2) - el_lims(1);

        n_azi = azi_interval / azi_res + 1;

        n_ele = ele_interval / el_res + 1;

        n_total = n_ele * n_azi;
    end
    
    % Print out inputs
    if debug
        disp(['Azimuth interval:  ', num2str(azi_lims(1)), ...
              ' to ', num2str(azi_lims(2)), ...
              'º;     Azimuth Resolution: ', num2str(azi_res)]);

        disp(['Elevation interval:  ', num2str(el_lims(1)), ...
              ' to ', num2str(el_lims(2)), ...
              'º;     Elevation Resolution: ', num2str(el_res)]);

        disp(['Total # of precoders: ', num2str(n_total)]);
    end
    
    
    gob_name = [num2str(arr_size(1)), '_', ...
                num2str(arr_size(2)), '_', ...
                num2str(azi_lims(1)), '_', ...
                num2str(azi_lims(2)), '_', ...
                num2str(azi_res), '_', ...
                num2str(azi_offset), '_', ...
                num2str(el_lims(1)), '_', ...
                num2str(el_lims(2)), '_', ...
                num2str(el_res), '_', ...
                num2str(el_offset), '_', ...
                'pol_', num2str(diff_polarisations)];

    
    
    
    
    fc = 1e9; c = physconst('LightSpeed'); lambda = c/fc;
    % Note: the computed precoders are for arrays with distance between
    %       elements of half wavelength. Therefore, if we are applying
    %       these precoders to physical (real) arrays, that are not
    %       operating at the centre frequency, then the precoders will be
    %       slightly less 'perfect' for those cases.
    normal_axis = 'x';
    inc_ele_resp = true;
    enable_pol = false;
    
    % Note: phase.URA takes ABSOLUTE LENGTHS, NOT ELECTRICAL! OF
    % COURSE, or else what would changing the frequency do?!
    array_abs_len = lambda * elect_ele_space;
    
    if use_taper
        gob_name = [gob_name, ...
                    '_taper_sll_', num2str(sll), ...
                    '_nbar_', num2str(nbar)];
    end
    
    
    if use_taper
        % Configure the taper
        twinz = taylorwin(arr_size(1),nbar,sll);
        twiny = taylorwin(arr_size(2),nbar,sll);
        taper = twinz * twiny.';
    end

    if diff_polarisations
        pol_indicator = 3;
        qd_3gpp_ele = qd_arrayant('3gpp-3d',...
                              1,...             % 1 element in vertical
                              1,...             % 1 element in horizontal
                              fc,...            % freq centre
                              pol_indicator,... % polarisation
                              0,...             % electrical down-tilt
                              elect_ele_space); % element spacing, [lambda]

        if debug
            % we need to separate the left and right elements!
            disp(['Orthogonal elements will be separated in +-45º ', ...
              'funny enough, they give the same precoder!']);
        end
        % Copy the left and right elements
        a_left =  phased.CustomAntennaElement(...
            'MagnitudePattern', 10 * log10(squeeze(qd_3gpp_ele.Fa(:,:,1)).^2));

        a_right =  phased.CustomAntennaElement(...
            'MagnitudePattern', 10 * log10(squeeze(qd_3gpp_ele.Fa(:,:,2)).^2));

        % We assume spacing in both directions is the same, 
        
        array_left_3gpp = phased.URA('Element', a_left, ...
                                     'Size', arr_size, ...
                                     'ArrayNormal', normal_axis, ...
                                     'ElementSpacing', array_abs_len);

        array_right_3gpp = phased.URA('Element', a_right, ...
                                      'Size', arr_size, ...
                                      'ArrayNormal', normal_axis, ...
                                      'ElementSpacing', array_abs_len);


        sv_obj_3gpp_left = phased.SteeringVector(...
                                'SensorArray', array_left_3gpp, ...
                                'IncludeElementResponse', inc_ele_resp, ...
                                'EnablePolarization', enable_pol);
                            
        sv_obj_3gpp_right = phased.SteeringVector(...
                                'SensorArray', array_right_3gpp, ...
                                'IncludeElementResponse', inc_ele_resp, ...
                                'EnablePolarization', enable_pol);  
        
    else
        pol_indicator = 1;
        qd_3gpp_ele = qd_arrayant('3gpp-3d',...
                              1,...             % 1 element in vertical
                              1,...             % 1 element in horizontal
                              fc,...            % freq centre
                              pol_indicator,... % polarisation
                              0,...             % electrical down-tilt
                              elect_ele_space); % element spacing, [lambda]
        if debug
            disp('The joint response is considered.'); %#ok<*UNRCH>
        end
        a_crossed_3gpp = phased.CustomAntennaElement(...
            'MagnitudePattern', 10 * log10(qd_3gpp_ele.Fa .^ 2)); %[dB]

        array_3gpp = phased.URA('Element', a_crossed_3gpp, ...
                                'Size', arr_size, ...
                                'ArrayNormal', normal_axis, ...
                                'ElementSpacing', array_abs_len);

        sv_obj_3gpp = phased.SteeringVector(...
                                'SensorArray', array_3gpp, ...
                                'IncludeElementResponse', inc_ele_resp, ...
                                'EnablePolarization', enable_pol);  

    end
      
    
    if plot_precoders
        f = figure;
        %f.WindowState = 'maximized';
        f.Position = [4 133 1274 464];

        if do_vid
            writerObj = VideoWriter(['GoB_', gob_name, '.avi']);
            writerObj.FrameRate = 2;

            % Open video writer and get a frame with the subtitles
            open(writerObj);
            frame = getframe(f); 
            writeVideo(writerObj, frame);
        end
        
    end
    
    % fix this for manual angles...
    precoders_array = zeros(n_azi, n_ele, prod(arr_size));
    
    if strcmp(manual_or_auto, 'auto1')
        el_values = el_lims(1):el_res:el_lims(2);
        azi_values = azi_lims(1):azi_res:azi_lims(2);
    end
    
    if strcmp(manual_or_auto, 'auto2')
        azi_values = [azi_lims(1):azi_res:-azi_offset ...
                      azi_offset:azi_res:azi_lims(2)];

        el_values = [el_lims(1):el_res:-el_offset ...
                     el_offset:el_res:el_lims(2)];

        if azi_offset == 0
            azi_values(find(azi_values==0, 1, 'first')) = [];
        end

        if el_offset == 0
            el_values(find(el_values==0, 1, 'first')) = [];
        end
    end
    
    
    
    
    if strcmp(manual_or_auto, 'auto1') || strcmp(manual_or_auto, 'auto2')
        i = 1;
        angles = {};
        idxs = {};
        for el_idx = 1:n_ele
            for azi_idx = 1:n_azi    
                angles{i} = [azi_values(azi_idx); el_values(el_idx)]; %#ok<AGROW>
                idxs{i} = [azi_idx, el_idx];
                i = i + 1;
            end
        end
    end
    
    
    
    for i = 1:n_ele * n_azi
        steer_angle = angles{i};
        if strcmp(manual_or_auto, 'auto1') || strcmp(manual_or_auto, 'auto2')
            azi_idx = idxs{i}(1);
            el_idx = idxs{i}(2);
        end
        
        if debug
            disp(['Computing steer angle: [', ...
                num2str(azi_values(azi_idx)), 'º, '...
                num2str(el_values(el_idx)), 'º]']);
        end

        if diff_polarisations
            % Compute precoder
            sv_left_3gpp = sv_obj_3gpp_left(fc, steer_angle);                    
            sv_right_3gpp = sv_obj_3gpp_right(fc, steer_angle);

            if use_taper
                sv_left_3gpp = sv_left_3gpp .* reshape(taper, [], 1);
                sv_right_3gpp = sv_right_3gpp .* reshape(taper, [], 1);
            end

            % Normalise
            sv_left_3gpp = sv_left_3gpp ./ norm(sv_left_3gpp);
            sv_right_3gpp = sv_right_3gpp ./ norm(sv_right_3gpp);

            precoders_array(azi_idx, el_idx, :) = sv_left_3gpp;

            if plot_precoders
                sgtitle(['Top row +45º, Bottom -45º, with ', ...
                         'azi: ', num2str(azi), ', el: ', num2str(el)]);

                s1 = subplot(2,3,1); %#ok<*NASGU>
                pattern(array_left_3gpp,fc,-180:180,steer_angle(2), ...
                        'Type','powerdb','Weights',sv_left_3gpp);
                s2 = subplot(2,3,2);
                pattern(array_left_3gpp,fc, steer_angle(1), -90:90,...
                        'Type','powerdb','Weights',sv_left_3gpp);
                s3 = subplot(2,3,3);
                pattern(array_left_3gpp,fc, ...
                        'Type','powerdb','Weights',sv_left_3gpp);

                s4 = subplot(2,3,4);
                pattern(array_right_3gpp,fc,-180:180,steer_angle(2), ...
                        'Type','powerdb','Weights',sv_right_3gpp);
                s5 = subplot(2,3,5);
                pattern(array_left_3gpp,fc, steer_angle(1), -90:90,...
                        'Type','powerdb','Weights',sv_right_3gpp);
                s6 = subplot(2,3,6);
                pattern(array_right_3gpp,fc, ...
                        'Type','powerdb','Weights',sv_right_3gpp);
                drawnow;

                if do_vid
                    frame = getframe(f); 
                    writeVideo(writerObj, frame);
                end
            end

            % IMPORTANT DISCOVERY:
            % THE PRECODERS ARE EXACTLY THE SAME FOR THE LEFT AND FOR
            % THE RIGHT ELEMENTS.
%                 if sv_right_3gpp ~= sv_left_3gpp %#ok<BDSCI>
%                     disp(['THE PRECODERS ARE DIFFERENT FOR ANGLE: ', ...
%                           num2str(steer_angle)]);
%                       % THIS HAS NEVER HAPPENNED!
%                 end

            % For the sake of speed, I'll comment the code that is
            % not needed for the precoder computation.
        else
            % Compute beamformer
            sv_3gpp = sv_obj_3gpp(fc, steer_angle);

            if use_taper
                sv_3gpp = sv_3gpp .* reshape(taper, [], 1);
            end

            % Normalise
            sv_3gpp = sv_3gpp ./ norm(sv_3gpp);

            precoders_array(azi_idx, el_idx, :) = sv_3gpp;

            if plot_precoders
                sgtitle(['Crossed dipole with ', ...
                         'azi: ', num2str(azi), ', el: ', num2str(el)]);

                s1 = subplot(1,3,1);
                pattern(array_3gpp,fc,-180:180,steer_angle(2), ...
                        'Type','powerdb','Weights',sv_3gpp);
                s2 = subplot(1,3,2);
                pattern(array_3gpp,fc,steer_angle(1),-90:90, ...
                        'Type','powerdb','Weights',sv_3gpp);
                s3 = subplot(1,3,3);
                pattern(array_3gpp,fc, ...
                        'Type','powerdb','Weights',sv_3gpp);
                zoom(2.2);

                drawnow;

                if do_vid
                    frame = getframe(f); 
                    writeVideo(writerObj, frame);
                end
            end
        end
    end
    
    
    % Save
    if save_precoders
        disp('Saving in a file...');
        save_file_name = ['precoders_', gob_name];
        save(save_file_name, 'precoders_array', 'azi_values', 'el_values');
    end
    
    if plot_precoders
        if do_vid
            close(writerObj);
        end
    end
end

