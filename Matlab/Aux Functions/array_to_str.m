function [s] = array_to_str(a)
    % returns a proper string for the array a
    s = '[';
    for i = 1:numel(a)
        s = [s, num2str(a(i)), ',']; %#ok<AGROW>
    end
    
    % Take the last comma out, and close brackets
    s = [s(1:end-1), ']'];
end

