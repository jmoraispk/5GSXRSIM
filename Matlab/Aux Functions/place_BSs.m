function [tx_pos] = place_BSs(n_tx, tx_pos, tx_special_pos, room_size, ...
                              tx_height, select_corners)
    
    % This function is responsible for placing the BS antennas in the room.
    % It only sets their centres, not their orientation.
    
    
    % Based on the combination of number of TXs and this variable, they
    % are placed in different places. 
    % 0 - no special places, just put them at the 'tx_pos' position
    % 1 - 'centre first': places BS at the centre first, and if there's
    %                     more than one, they are placed in the corners
    %                     of the room. Requires 'room_size' variable!
    % 2 - 'corners first': places BS at the corners first. If there's more
    %                      than 4, puts at the centre.
    % Note: max number of TXs (BSs) to be placed automatically is 5!
    %       For n_tx > 5, the positions need to be set in tx_pos.
    
    
    % Finally, select_corners is a list with as much as 4 elements,
    % prioritizing the corners, knowing that the normal priority is 
    % starting at the origin and following counter-clock-wise
    
    
    if tx_special_pos == 0
        return % keep the same tx_pos
    end
    
    if n_tx > 5
        error('Maximum number of tx supported is 5');
    end
    
    if n_tx <= 0
        error("What's the point?")
    end
    
    if tx_special_pos > 2
        error('Only 3 possible values for tx_special_pos: 0, 1 or 2');
    end
    
    % From here on, tx is either 1 or 2.
    tx_pos = [];
    antennas_left_to_place = n_tx;
    
    centre_of_table_pos = [room_size(1)/2; room_size(1)/2];
    
    
    if tx_special_pos == 1 || n_tx == 5
        % place one at the centre.
        tx_pos = [centre_of_table_pos; tx_height];
        antennas_left_to_place = antennas_left_to_place - 1;
    end
    
    if n_tx == 1 && tx_special_pos == 1
        return
    end
    
    % The corners of the room:
    room_corners = [ 0,                       0,         tx_height;
                     room_size(1),            0,         tx_height;
                     room_size(1), room_size(2),         tx_height;
                     0,            room_size(2),         tx_height;]';
                 
    if ~isempty(select_corners)
        room_corners = room_corners(:, select_corners);
    end
    if tx_special_pos == 2 || n_tx > 1
        % Start placing the antennas around the room
        for i = 1:antennas_left_to_place
            tx_pos = [tx_pos room_corners(:, i)]; %#ok<AGROW>
        end
    end
    
end
