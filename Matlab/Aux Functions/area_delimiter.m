function [ar] = area_delimiter(x_data, y_data, sep_snap, last_snap, ...
                               lims, fig_idx)
    %Returns the data to plot in order to achieve the nice separations
    %Like QuaDRiGa example 2
    
    % NOTE 1: this function needs a figure that is being holded on and that
    %already has the axis defined, so that the figure doesn't change
    %depending on the values of the data being ploted after axis definition
    
    % NOTE 2: this function only works for only positive or only negative
    % y_data values. Or else there will be a change in color. See below 
    % what the 'area' function does.
    
    
    % IF LIMS, then maintain the limits of the previous graph, and
    % tune down the opacity
    if lims
        % Get figure limits before
        x_lim = xlim();
        y_lim = ylim();
    end
    
    %decide if y_data is only positive or only negative
    if mean(y_data) > 0
        aux = 1;
    else
        aux = -1;
    end
    aux = 1;
    % for positive data, we want the auxiliar data to be -INF 'normally' 
    % (so white) and INF the other times (green) and the opposite for
    % negative data
    
    ar   = ones(1, last_snap) * (-1e9) * aux;                  % Shading of events
    
    for i = 1:length(sep_snap)
        if i == length(sep_snap)
            fill_until = last_snap;
        else
            fill_until = sep_snap(i+1);
        end
        
        if mod(i,2)
            ar(sep_snap(i): fill_until) = 1e9 * aux;
        end
    end

    % Area shading (Shades area between function and x axis!)
              %x    %y   %color (greenish)        %no line separating
    if fig_idx
        figure(fig_idx)
    end
    a = area(x_data,ar, -1e9, 'FaceColor',[0.7 0.9 0.7],'LineStyle','none'); 
    
    % stupid trick to do reduce face alpha without setting limits
    if isempty(lims) && isempty(fig_idx)
        a.FaceAlpha = 0.25;
    end
    
    if lims
        a.FaceAlpha = 0.5;
        % Set the limits back to what they were.
        xlim(x_lim);
        ylim(y_lim);
    end
end

