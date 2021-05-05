function [max_snap] = calc_max_snap(tracks)
    %CALC_USABLE_INTERVAL 
    % From the different tracks of each receiver, it calculates
   
    max_snap = inf;
    for i=1:length(tracks)
        snaps_for_user_i = size(tracks(i).positions,2);
        if (snaps_for_user_i < max_snap) && (snaps_for_user_i > 1)
            max_snap = snaps_for_user_i;
        end
    end
end

