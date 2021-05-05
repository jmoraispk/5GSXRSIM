function s = get_curr_time_str()

    % Returns a string with the time, that can be used to create folders
    % i.e. doesn't have the character ':'
    
    s = char(datetime('now'));
    s(length(s)) = 's';
    s(end-2) = 'm';
    s(end-5) = 'h';
    
end