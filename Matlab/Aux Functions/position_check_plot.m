function [] = position_check_plot(phy_usr_pos, vir_usr_pos, tx_pos, ...
                                  n_camera_streams, cam_pos)
        
    figure;
    scatter3(phy_usr_pos(1,:), phy_usr_pos(2,:), phy_usr_pos(3,:), ...
             100, 'r','s', 'filled');
    hold on;
    scatter3(vir_usr_pos(1,:), vir_usr_pos(2,:), vir_usr_pos(3,:), ...
             100, 'b', 's', 'filled');
    scatter3(tx_pos(1), tx_pos(2), tx_pos(3), 100, 'filled');
    if n_camera_streams ~= 0
        scatter3(cam_pos(1,:), cam_pos(2,:), cam_pos(3,:), ...
             100, 'k', 's', 'filled');
    end

    if n_camera_streams > 0
        legend({'phy', 'vir', 'tx', 'cam'});
    else
        legend({'phy', 'vir', 'tx'});
    end

    %get the axis limits and set the smaller to the biggest
    y_lim = ylim;
    x_lim = xlim;
    if (y_lim(2) - y_lim(1)) > (x_lim(2) - x_lim(1))
        xlim(y_lim)
    else
        ylim(x_lim)
    end
    hold off;
    pls_labels();
end

