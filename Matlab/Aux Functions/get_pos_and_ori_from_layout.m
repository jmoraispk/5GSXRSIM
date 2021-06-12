function [pos_backup, ori_backup, initial_pos_backup] = ...
                                             get_pos_and_ori_from_layout(l)
    n_rx = l.no_rx;
    pos_backup = zeros(n_rx, 3, l.rx_track(1,1).no_snapshots);
    ori_backup = zeros(n_rx, 3, l.rx_track(1,1).no_snapshots);
    initial_pos_backup = zeros(n_rx, 3);
    
    for rx_idx = 1:n_rx
        % each positions or orientations vector is [3 x n_tti]
        pos_backup(rx_idx, :, :) = l.rx_track(rx_idx).positions;
        ori_backup(rx_idx, :, :) = l.rx_track(rx_idx).orientation;
        initial_pos_backup(rx_idx, :) = l.rx_track(rx_idx).initial_position;
    end
end