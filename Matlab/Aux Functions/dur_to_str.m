function [d_str] = dur_to_str(d)
    
    d_str = char(d);
    d_str(3) = 'h';
    d_str(6) = 'm';
    d_str(end+1) = 's';
end

