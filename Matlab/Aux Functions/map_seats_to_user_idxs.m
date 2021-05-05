function [idxs] = map_seats_to_user_idxs(seats, phy_disp, vir_disp)

    idxs = zeros(size(seats));
    
    for i = 1:length(seats)
        seat = seats(i);
        
        % try to find in one of the lists
        if find(phy_disp == seat)
            idxs(i) = find(phy_disp == seat);
        elseif find(vir_disp == seat)
            idxs(i) = length(phy_disp) + find(vir_disp == seat);
        else
            disp("Couldn't find seat in either user disposition!")
            return
        end
    end
    
end