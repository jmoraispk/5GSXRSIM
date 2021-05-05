function [avg_drop] = avg_drop(a)
    %avg_drop 
    % Takes an array of sorted values (biggest to smallest)
    % computes the average of their differences
    
    n = length(a);
    diffs = zeros(n-1,1);
    
    for i = 1:n-1
        diffs(i) = a(i) - a(i+1);
    end
    
    avg_drop = mean(diffs);
end

