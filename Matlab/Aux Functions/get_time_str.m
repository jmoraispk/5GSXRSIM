function [t_str] = get_time_str(short_or_full)
    if nargin < 1 
        short_or_full = 'short';
    end
    %returns date and time in string
    time_now = clock;
    
    
    day_and_time = [num2str(round(time_now(3))), 'd', num2str(time_now(4)), 'h',...
                   num2str(time_now(5)), 'm', num2str(round(time_now(6))), 's'];
               
    if strcmp(short_or_full, 'complete')
        year_month = [num2str(time_now(1)), 'Y', num2str(time_now(2)), 'M'];
        t_str = [year_month day_and_time];
    else
        t_str = day_and_time;
    end
end

