function [] = rotate_all_elements(ant, ant_conf, ang, axis, ...
                                  rotate_in_place, diff_pol)
    % Rotates all elements in the array
    n = ant_conf(1) * ant_conf(2);
    
    if diff_pol
        n = n * 2;
    end
    
    if rotate_in_place
        prev_pos = ant.element_position;
    end
    
    ant.rotate_pattern(ang, axis, 1:n);
    
    if rotate_in_place
        ant.element_position = prev_pos;
    end
end

