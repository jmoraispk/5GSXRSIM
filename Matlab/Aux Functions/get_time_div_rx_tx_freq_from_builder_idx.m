function [time_div, rx_idx, tx_idx, freq_idx] = ...
          get_time_div_rx_tx_freq_from_builder_idx(builder_idx, n_rx, n_tx, n_freq)
    
    time_div = floor(builder_idx / (n_rx * n_tx * n_freq)) + 1;
    if mod(builder_idx, n_rx * n_tx * n_freq) == 0
        time_div = time_div - 1;
    end
    leftover_time = builder_idx - (n_rx * n_tx * n_freq) * (time_div - 1);
    
    freq_idx = floor(leftover_time/ (n_rx * n_tx)) + 1;
    if mod(leftover_time, n_rx * n_tx) == 0
        freq_idx = freq_idx - 1;
    end
    leftover_freq = leftover_time - (n_rx * n_tx) * (freq_idx - 1);
    
    tx_idx =   floor((leftover_freq - 1)/ n_rx) + 1;
    rx_idx =     mod(builder_idx, n_rx) ;
                    
    if rx_idx == 0
        rx_idx = n_rx;
    end
    
end