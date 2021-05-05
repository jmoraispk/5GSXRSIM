function [] = point_antenna_to_target(antenna_track, antenna, target)

    % Makes necessary rotations such that the antenna points at the target
    % Note: the modifications are computed assuming the antenna is in it's
    %       neutral position, which is pointing along the x axis.
    %antenna.rotate_pattern(90, 'x');
    %antenna.rotate_pattern(90, 'y');
    
    ant_pos = antenna_track.initial_position;    
    % First, we apply the elevation rotation
    % This allows us to apply it only on one axis, either x or y
    % Depending on which the antenna is oriented.

    % BOTTOM LINE: Rotate on y, all the antennas face x by default!
    dist_from_above = sqrt((ant_pos(1) - target(1))^2 + ...
                           (ant_pos(2) - target(2))^2);
     
    % Compute the symmetrical of elevation.
    y_rot_ang = atan(dist_from_above / (ant_pos(3) - target(3)));
    
    
    
    y_rot_ang_deg = rad2deg(y_rot_ang);
    
    if y_rot_ang_deg == 0
        % the antenna may be exactly below or above the target point!
        if ant_pos(3) > target(3)
            y_rot_ang_deg = 90;
        elseif ant_pos(3) < target(3)
            y_rot_ang_deg = -90;
        end
        % otherwise, it's suppose to be 0, it's at the same height.
    end
    
%     disp(['Would rotate downwards: ', num2str(y_rot_ang_deg), 'º']);
    antenna.rotate_pattern(y_rot_ang_deg, 'y');
    
    
    % Rotation around z (Azimuth)
    azi_ang = calc_2Dangle_from_pos(ant_pos(1:2), target(1:2));
    azi_ang_deg = rad2deg(azi_ang);
    antenna.rotate_pattern(azi_ang_deg, 'z');
    
%     disp(['Would rotate on azimuth:', num2str(azi_ang_deg), 'º']);
end

