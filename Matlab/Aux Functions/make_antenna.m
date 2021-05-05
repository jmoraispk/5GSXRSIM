function [ant] = make_antenna(type, freq, diff_orthogonal_polarisation, ...
                              elements_config, element_spacing, ...
                              hybrid_bf, subarray_size, subarray_response)
	%Creates one of the specified antennas for a given frequency
    % patch, (half-wave)dipole, omni, rectangular
    debug = 0;
    
    type = char(type); % a fix for cells with strings
    
    if debug
        disp(['Type: ', type]);
    end
    
    switch type
        case 'omni'
            ant = qd_arrayant();
            ant.center_frequency = freq;
        case 'patch'
            ant = qd_arrayant('patch');
            ant.center_frequency = freq;
        case 'dipole'
            ant = qd_arrayant('half-wave-dipole');
            ant.center_frequency = freq;
        case 'array'                      % elements
            
            
            % Choose to get the response of 1 cross-polarised element, or
            % the response of each of the orthogonal elements that comprise
            % it
            
            % If Hybrid BF is used, we need to choose if we want to get
            % responses from the individual elements in the subarray or not
            if ~diff_orthogonal_polarisation
                if ~(hybrid_bf && subarray_response)
                    pol_indicator = 1;
                else
                    pol_indicator = 4;
                end
            else
                if ~(hybrid_bf && subarray_response)
                    pol_indicator = 3;
                else
                    pol_indicator = 6;
                end
            end
    
            % Pol_indicator(Din input, equivalent to K in 3GPP spec p21 38.901)
            
            % Pol_indicator = 1:
            % 1 coeff per cross-polarised element
            
            % Pol_indicator = 3:
            % 2 coeffs per cross-polarised element, one for each polarisation
            
            % Pol_indicator = 4:
            % 1 coeff per sub-array made out of M cross-polarised elements
            
            % Pol_indicator = 6:
            % 2 coeffs per sub-array made out of M SINGLE-polarised elements
            
            % Pol_indicator 2 and 4 are only rotated versions of 3 and 6,
            % respectively. Thus, not interesting.
            
            
            if hybrid_bf == 0
                ant = qd_arrayant( '3gpp-mmw', ...
                    elements_config(1), ... % # elements in the vertical
                    elements_config(2), ... % # elements in the horizontal
                    freq, pol_indicator, ...% freq and polalisation ind. 
                    0,element_spacing, ...  % electric downtilt and element spacing
                    1,1);                   % num of panels and spacings (defaults used)
            else
                
                n_subarrays(1) = elements_config(1) / subarray_size(1);
                n_subarrays(2) = elements_config(2) / subarray_size(2);
                if debug
                    disp(['Creating a [', ...
                      num2str(subarray_size(1)), ',', ...
                      num2str(subarray_size(2)), '] subarray', ...
                      ' and [', num2str(n_subarrays(1)), ...
                      ', ', num2str(n_subarrays(2)), ...
                      '] # of subarrays. Pol indicator: ', ...
                      num2str(pol_indicator), '. If pol_indicator==6, ', ...
                      'expect ', num2str(prod(n_subarrays) * 2), ...
                      ' coeffs.']); %#ok<*UNRCH>
                end
                ant = qd_arrayant( '3gpp-mmw', ...
                    subarray_size(1), ...           % # vertical elements in sub-array
                    subarray_size(2),  ...          % # horizontal elements in sub-array
                    freq, pol_indicator, ...        % freq and polalisation ind. 
                    0, element_spacing, ...         % electric downtilt and element spacing
                    n_subarrays(1), n_subarrays(2));% # of subarrays vertically and horizontally 
            end
                
        otherwise
            error('Type should be "omni", "patch", "dipole" or "array"');
    end
    
    % Set values to single (+30% performance increase, they say)
    ant.single()
end

