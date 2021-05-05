function [z_interp] = interpolate_orientation_respecting_table(z_rot1, ...
                                                               z_rot2, ...
                                                               is)
    % 'is' is the number of samples in between rot1 and rot2.


    if (z_rot1 < -pi/2 && z_rot2 > pi/2) || ...
       (z_rot1 > pi/2 && z_rot2 < -pi/2)
        if (z_rot1 < -pi/2 && z_rot2 > pi/2)
            z_rot1 = z_rot1 + pi;
            z_rot2 = z_rot2 - pi;
        else
            z_rot1 = z_rot1 - pi;
            z_rot2 = z_rot2 + pi;
        end

        % interpolate
        z_interp = linspace(z_rot1, z_rot2 , is);

        % modify z_rot to match the proper values.
        for idx = 1:is
            if z_interp(idx) > 0
                z_interp(idx) = z_interp(idx) - pi;
            else
                z_interp(idx) = z_interp(idx) + pi;
            end
        end
    elseif (z_rot1 < 0 && z_rot2 > 0 && ...
            (z_rot2 - z_rot1) > pi) || ...
            (z_rot1 > 0 && z_rot2 < 0 && ...
            (z_rot1 - z_rot2) > pi)

        % will be used to go back to the proper angles.
        z_rot1_original = z_rot1;
        z_rot2_original = z_rot2;

        % make transformation
        if z_rot1 < 0
            z_rot1 = z_rot1 + 2*pi;
            situation = 1;
        else
            z_rot2 = z_rot2 + 2*pi;
            situation = 2;
        end

        % interpolate
        z_interp = linspace(z_rot1, z_rot2 , is);

        % modify z_rot to match the proper values.
        for idx = 2:is
            if situation == 1
                z_interp(idx) = wrapToPi(z_rot1_original - ...
                                         (z_interp(1) - z_interp(idx)));
            else
                % Actually, this never goes here because of
                % the order we are switching between
                % virtual users. If we changed counter-
                % clock-wise we'd enter in this else.

                % it is implemented for completeness sake.
                z_interp(idx) = wrapToPi(z_rot1_original + ...
                                         (z_interp(1) - z_interp(idx)));
            end
        end
        if situation == 1
            z_interp(1) = z_rot1_original;
        else
            z_interp(end) = z_rot2_original;
        end
    else
        z_interp = linspace(z_rot1, z_rot2 , is);
    end
end