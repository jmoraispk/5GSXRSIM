function [az, el] = get_view(view_word)
    %GET_VIEW returns azimuth and elevation values for certain keywords
    
    % to know a certain view of a figure do: ax = gca; ax.View
    
    switch(view_word)
        case {"up", "xy"}
            az = 0; el = 90;
        %%%%%%%%%%%%%%%%%%%%%%%%
        case "yx"
            az = 90; el = 90;
        %%%%%%%%%%%%%%%%%%%%%%%%
        case "side"
            az = 0; el = 0;
        case "xz"
            az = 0; el = 0;
        %%%%%%%%%%%%%%%%%%%%%%%%
        case "front"
            az = 90; el = 0;
        case "yz"
            az = 90; el = 0;
        %%%%%%%%%%%%%%%%%%%%%%%%    
        case "back"
            az = 90; el = 0;
        %%%%%%%%%%%%%%%%%%%%%%%%
        case "normal"
            az = -33; el = 30;
        case "normal2"
            az = 150; el = 30;
        case "normal3"
            az = 30; el = 20;
        otherwise
            % view_word is an array and we just assign it to az and el.
            view_cells = regexp(view_word, ' ', 'split');
            az = str2double(view_cells{1}); 
            el = str2double(view_cells{2});
    end
end

