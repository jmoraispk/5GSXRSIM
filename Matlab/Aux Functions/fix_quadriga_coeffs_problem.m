function [] = fix_quadriga_coeffs_problem(qd_antenna_handle, ant_conf, ...
                                          diff_pol)

    % This function fixes the Quadriga coefficients ordering problem
    
    % This way, the precoders from Phased Array and Quadriga will have
    % the exact same effect, in terms of the direction to which they point
    % beams.
    
    qd_antenna_handle.rotate_pattern(180, 'z');
    
    % And rotate each element back such that the other rotations have
    % the desired effects, and the radiation patterns are properly oriented
    % This rotates the elements in place.
    rotate_all_elements(qd_antenna_handle, ant_conf, 180, 'z', 1, diff_pol)
end

