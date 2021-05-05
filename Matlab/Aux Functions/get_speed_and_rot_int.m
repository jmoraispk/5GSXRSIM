function [speed, rot_int] = get_speed_and_rot_int(mvnt_idx)
    % Returns the speed and the rotation interval 
    %based on the mvmnt indexes
    
    len = length(mvnt_idx);
    speed = zeros(1,len);
    rot_int = zeros(1,len);
    
    for i = 1:len
        speed(i) = mvnt_idx(i) * 0.1;
        rot_int(i) = 1.5 - 0.2 * mvnt_idx(i);
    end
end

