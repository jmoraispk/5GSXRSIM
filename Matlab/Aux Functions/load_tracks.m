function [] = load_tracks(l, filename)
    % Load layout with created tracks
    l_aux = load(filename).l_aux;
    
    for k = 1:l.no_rx
        l.rx_track(1,k) = l_aux.rx_track(1,k).copy;
    end
end