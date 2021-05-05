function [diffs] = calculate_interval_diffs(a, interval)
    % computes the differences between adjacent elements (interval=1)
    % or elements spaced the interval 

    diffs = a(1+interval:end) - a(1:end-interval);
end

