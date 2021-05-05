function [c_stitched] = stitch_channels(channels)
    %
    % Joins multiple channels together. Appends all the channel parameters,
    % with the only exception of the (pg) and (pg_parset) parameters.
    
    % These can be joined as well by doing what get_channels.m does to
    % compute them in the first place. 
    % To the best of my knowledge, they are only used for Singular Value
    % Scalling. The path gain (pg) only exists if non-3GPP baseline is 
    % selected, which is not the case for spherical waves.
    
    % This function joins as many channels as given, appending positions,
    % coefficients and delays, and keeping the rest of the information 
    % from the first channel (c1).
    % Provided all channels are generated from a builder 
    % with the same Large and Small Scale Fading parameters.
    
    size_c1 = size(channels(1).coeff);
    
    % the total number of snapshots is the summer
    total_snap = sum(channels(1).no_snap);
    
    % the same number of tx and rx is assumed
    result_size = [size_c1(1:3), total_snap];
    
    coeffs = zeros(result_size);
    delays = zeros(result_size);
    positions = zeros(3, total_snap);
    
    
%     % Build coeffs and delays matrices
%     coeffs(:,:,:,1:size_c1(4)) = channels(1}.coeff;
%     coeffs(:,:,:,size_c1(4)+1:end) = channels(2}.coeff;
% 
%     delays(:,:,:,1:size_c1(4)) = channels(1}.delay;
%     delays(:,:,:,size_c1(4)+1:end) = channels(2}.delay;
% 
%     % Build positions (we assume tx static!)
%     positions(:,1:size_c1(4)) = channels(1}.rx_position;
%     positions(:,size_c1(4)+1:end) = channels(2}.rx_position;

    accumulated_counter = 0;
    for i = 1:length(channels)
        % Idxs for appending matrices
        n_to_add = size(channels(i).coeff, 4);
        
        % to account for the 1:5 5:10 phenomena where 5 goes in twice.
        first_idx = accumulated_counter + 1;
        last_idx = accumulated_counter + n_to_add;
        
        % Build coeffs and delays matrices
        coeffs(:,:,:, first_idx:last_idx) = channels(i).coeff;
        delays(:,:,:, first_idx:last_idx) = channels(i).delay;

        % Build positions (we assume tx static, so only rx)
        positions(:, first_idx:last_idx) = channels(i).rx_position;
        
        accumulated_counter = accumulated_counter + n_to_add;
    end
    
    c_stitched = channels(1).copy();
    c_stitched.coeff = coeffs;
    c_stitched.delay = delays;
    c_stitched.rx_position = positions;
    %no_snap is set automatically
    
end

