function [phy_pos, vir_pos, cam_pos] = ...
               room_arrangement(room_type, room_size, rect_sides, ...
               room_capacity, n_phy, n_virtual, n_cameras_per_user, ...
               cam_params, chair_dist, phy_user_disposition, ...
               vir_user_disposition, heights, r_table, r_users)

    % room_arrangement Distributes user and camera positions in a room
    
    % type_aux - used to specify the room further. For instance, when the
    % room is rectangular, it gives the ratio between users at each side of
    % the table
    
    % n_total - is the total table capacity (NOTE: there can be less people
    % around the table than the table can seat)
    
    % n_phy & n_virtual - the number of people, respectively,  present 
    % and not present 
    
    % n_cameras_per_user - number of cameras per user these will be placed
    % a certain distance from the user, on top of the table.
    
    % cam_params - is an array with camera parameters. 
    % The first and second values are d_u and d_s, the camera distances;
    % The third value is:
    % ------------------THIS IS NOT IMPLEMENTED-------------------
    % if cameras should be placed around the room. The 
    % room is either square (if type = round or square) or rectangular.
    % The cameras are distributed in a uniform manner around the corners
    % and edges of a room. It starts on the corners (if param(1) is even), 
    % or on edges (param(1) is odd). Since the room always has 4 corners 
    % and 4 edges, the parameter should be a multiple of 4 and that 
    % multiple is the number of cameras around the room. If it's odd, 
    % then 1 is subtracted. E.g.:
        % 0 - no cameras besides the user-specific cameras
        % 4 - 4 cameras on the corners (1 per corner)
        % 5 - 4 cameras on edges       (1 per edge)
        % 8 - 1 per corner, 1 per edge
        % 9 - 2 per edge
        % 12 - 1 per corner, 2 per edge
        % 13 - 3 per edge
        % ....
    % The rest of the parameters of the array are a way of knowing how far
    % from the walls or from the table the cameras should be. 
    % ------------------THIS IS NOT IMPLEMENTED-------------------
    
    % room_config - is used to 
    
    % chair_dist - will be used to calculate positions and table size
    
    % camera_disposition - to define the camera placement strategy
    
    % user_disposition - indexes of the present people, 
    % hence len = n_present.
    
    % heights - 2 element array, respectively, user and camera heights
    
    %NOTE: the botom left of the room will be at the origin
    
    if (room_capacity < 2) || (n_phy < 1) || (n_virtual < 1) || ...
       ((n_phy + n_virtual) > room_capacity)
        error(['room_capacity must be more than 2 and there must be at',...
               ' least one virtual and one physical user, and the sum', ...
               'of virtual and physical users should be less or equal', ...
               'to the total number of users']);
    end
    
    if n_phy == room_capacity
        error('why using vr?');
    end
    
    if isempty(chair_dist)
        chair_dist = 1;
    end
    
    if isempty(cam_params)
        cam_params = 0;
    end
    
    if isempty(n_cameras_per_user)
        n_cameras_per_user = 2;
    end
    
    if n_cameras_per_user ~= 2 && n_cameras_per_user ~= 0 ...
                               && n_cameras_per_user ~= 1
        error('Only 2 or no cameras per users are supported/implemented');
    end
    
    if length(heights) ~= 2
        error('heights must be a 2 element array with user and camera heights');
    end
    
    if strcmp(room_type, 'round') && (room_capacity == 2)
        error('Select rectangular with side one for this effect');
    end
    
    
    % Adjust Phy and virtual user's dispositions
    
    if ~isstring(phy_user_disposition)
        phy_user_disposition = phy_user_disposition + 1;
    end
    if ~isstring(vir_user_disposition)
        vir_user_disposition = vir_user_disposition + 1;
    end
    
    
    % Check user disposition
    if isempty(phy_user_disposition)
        random_disposition = 1;
    elseif strcmp(phy_user_disposition, 'uniform')
        dispose_uniformly = 1;
        random_disposition = 0;
    else
        %distribute accordingly to the disposition
        
        if length(phy_user_disposition) ~= n_phy
            error(['Disposition has wrong dimensions. It must be an array', ...
                   ' with as many entries as present users.']);
        end
        
        if (max(phy_user_disposition) > room_capacity) || ...
           (min(phy_user_disposition)) < 1
            error('PHY user disposition Indexes are not supported');
        end
            
        random_disposition = 0;
        dispose_uniformly = 0;
    end
    
    
    if ~isempty(vir_user_disposition)
        % check if it's 'uniform' or the actual disposition
        if ~isstring(vir_user_disposition) && ...
           ~ischar(vir_user_disposition) && ...
            length(vir_user_disposition) ~= n_virtual
            error('Every virtual user should have a defined position');
        end
        
        
        %check if it's supported
        if (max(vir_user_disposition) > room_capacity) || ...
           (min(vir_user_disposition)) < 1
            error('PHY user disposition Indexes are not supported');
        end
        
        %check if there are collisions with the phy user disposition
        if ~isempty(intersect(phy_user_disposition, vir_user_disposition))
            error('A physical and a virtual user can'' seat in the same place');
        end
        
        %the check above may pass if phy_user_disposition is 'uniform'
        if strcmp(phy_user_disposition, 'uniform')
            warning(['The indexes for the virtual users were not ',...
                     'applied because the physical users uniform ', ...
                     'distribution has priority. Instead, the virtual', ...
                     'the virtual user''s disposition was chosen ', ...
                     'randomly. Visualization is recommended!']);
        end
    end
    
    
    centre = [room_size(1) / 2; room_size(1) / 2; heights(1)];
    
    % Determine the positions of the chairs
    switch room_type
        case {'round', 'circular'}
            % radius in order to have a chair_distance  between chairs
            if isempty(r_table) && isempty(r_users)
                r = chair_dist / 2 / sin(2 * pi / room_capacity);
            else
                if isempty(r_users)
                    % give an extra distance for the users. The table radius is not
                    r = r_table + 0.2;
                else
                    r = r_users;
                end
            end
            
            % the centre will be a radius above the first user
            
            positions(:,1) = centre - [0;r;0];
            for k = 1:(room_capacity-1)
                %angle between the points First user, centre, current_user
                %-pi/2 is the angle of the first user 
                ang = k * (2 * pi / room_capacity) - pi/2; 
                positions(:,k+1) = centre + [r * cos(ang), r * sin(ang), 0]';
            end
            
        case {'rectangular', 'square'}
            
            if strcmp(room_type, 'rectangular') 
                %if rectangular
                %don't check, use the provided sides
%                 if (rect_sides(1) * rect_sides(2)) ~= room_capacity
%                     error(['When rectangular is selected, type_aux should',...
%                            ' be an array with the number of users along ',...
%                            'the horizontal size on the first position, and',...
%                            ' the number of users along the vertical in the 2nd']);
%                 end

                n_hor = rect_sides(1);
                n_ver = rect_sides(2);
            else
                %if square
                side = ceil(sqrt(room_capacity));
                n_hor = side;
                n_ver = side;
            end
            
            
            % Find the position of the first user 
            positions(:,1) = centre - [(rect_sides(1) + 1)/2; ...
                                       (rect_sides(2) + 1)/2; ...
                                       0];
            
            %strategy: make a bigger square and remove corners
            n_hor = n_hor + 1;
            n_ver = n_ver + 1;
            
            % build the position vector
            for i = 1:4
                last_pos = positions(:, end);
                if mod(i,2) %horizontal
                    if i < 2
                        aux = 1;
                    else
                        aux = -1;
                    end
                    positions = [positions ...
                            [last_pos(1) + aux .* (1:n_hor) .* chair_dist ; ...
                             last_pos(2) * ones(1, n_hor); ...
                             last_pos(3) * ones(1, n_hor)] ];
                else %vertical
                    if i < 3
                        aux = 1;
                    else
                        aux = -1;
                    end
                    positions = [positions ...
                            [last_pos(1) * ones(1, n_ver); ...
                             last_pos(2) + aux .* (1:n_ver) .* chair_dist ; ...
                             last_pos(3) * ones(1, n_ver)] ]; %#ok<*AGROW>
                end
            %disp(positions)
            end
            
            %remove corners
            positions(:, end) = [];
            positions(:, 1) = [];
            positions(:, 2*n_hor + n_ver) = [];
            positions(:, n_hor +  n_ver) = [];
            positions(:, n_hor) = [];
            

        otherwise
            error('Unknown type. Choose: round, rectangular or square');
    end
    
    %scatter3(positions(1,:), positions(2,:), positions(3,:))
    
    %Pick the positions of the present users
    if random_disposition
        % distribute randomly - pick n_present places out of the available
        phy_idxs = randsample(1:size(positions,2), n_phy);
        
    elseif dispose_uniformly
        % distribute the users uniformly
        if n_phy == 1
            phy_idxs = 1;
        else
            phy_idxs = round(linspace(1,room_capacity, n_phy+1));
            phy_idxs(end) = []; %take the last out, will be adjacent to 1
        end
    else
        phy_idxs = phy_user_disposition;
    end
    
    phy_pos = positions(:, phy_idxs);
    positions_updated = positions; positions_updated(:, phy_idxs) = [];
    if ~isempty(vir_user_disposition)
        if strcmp(phy_user_disposition, 'uniform')
            %uniform has priority, apply randomly instead.
            vir_idxs = randsample(1:size(positions_updated,2), n_virtual);
            vir_pos = positions_updated(:, vir_idxs);
        else
            vir_pos = positions(:, vir_user_disposition);
        end
    else
        vir_idxs = randsample(1:size(positions_updated,2), n_virtual);
        vir_pos = positions_updated(:, vir_idxs);
    end
    
    cam_pos = [];
    
    
    % Camera position from here onwards
    
    if n_cameras_per_user == 0
        return
    end
    
    
    %Given the user positions, place the cameras
    % for 2 cameras, let's assume they were placed a distance d_s
    %place each camera a rectangle triangle away, where the hipotnuse
    %is the distance from the camera to the user, d_u is the distance 
    %from the user to the centre of the cameras and d_s  is the remaining 
    %side, which is half of the distance between the two cameras
    d_u = cam_params(1);
    d_s = cam_params(2);
    
    
    for u = 1:n_phy
        % Find one end of the vector
        switch room_type
            case 'round'
                % in a round table, the perpendicular to the user is a radius
                % so we can use the actual centre of the table
                point_in_user_perpendicular = repmat(centre, [1, n_phy]);
                
            case {'rectangular', 'square'}
                %in a square table, the perpendicular to the user is a 
                %perpendicular to the edge of the table the user is seated at.
                % we can find this by the phy_idx
                
                if phy_idxs(u) <= rect_sides(1)
                    %  bottom edge, point is along vertical parallel to +y
                    point_in_user_perpendicular = phy_pos + [0;1;0];
                elseif (phy_idxs(u) > rect_sides(1)) && ...
                       (phy_idxs(u) <= (rect_sides(1) + rect_sides(2))) 
                    % right edge, point is along horizontal parallel to -x
                    point_in_user_perpendicular = phy_pos + [-1;0;0];
                elseif (phy_idxs(u) > (rect_sides(1) + rect_sides(2))) && ...
                       (phy_idxs(u) <= (2*rect_sides(1) + rect_sides(2)))
                    % top edge, point is along vertical parallel to -y
                    point_in_user_perpendicular = phy_pos + [0;-1;0];
                else  
                    %left edge, point is along horizontal parallel to +x
                    point_in_user_perpendicular = phy_pos + [1;0;0];
                end

        end
        
        %find the vector that points from the physical user to the centre
        v1 = point_in_user_perpendicular(1:2, u) - phy_pos(1:2,u);

        %find an orthogonal vector
        v2 = [- v1(2); v1(1)];

        % make them unitary so it is as long as a constant scaling
        v1 = v1 ./ norm(v1);
        v2 = v2 ./ norm(v2);

        %use these vectors to find the camera positions
        if n_cameras_per_user == 1
            cam_pos = [cam_pos [phy_pos(1:2, u) + v1 * d_u; heights(2)]];
        else
            cam1 = [phy_pos(1:2, u) + v1 * d_u + v2 * d_s; ...
                    heights(2)];

            cam2 = [phy_pos(1:2, u) + v1 * d_u - v2 * d_s; ...
                    heights(2)];

            cam_pos = [cam_pos cam1 cam2];
        end
    end
        
    %Pick Camera Positions
    if cam_params(3)
        %place cameras around the room
    end
    
end

