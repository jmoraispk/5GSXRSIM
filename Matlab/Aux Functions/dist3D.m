function [ dist ] = dist3D(ant_rx_pos, ant_tx_pos)

%%%calculates the 3D distance between 2 points (TX AE and RX AE
%%%coordinates)

dist = sqrt( (ant_rx_pos(1, :) - ant_tx_pos(1, :) ) .^ 2 +...
    ( ant_rx_pos(2, :) - ant_tx_pos(2, :) ) .^ 2 +...
    ( ant_rx_pos(3, :) - ant_tx_pos(3, :) ) .^ 2 );
end