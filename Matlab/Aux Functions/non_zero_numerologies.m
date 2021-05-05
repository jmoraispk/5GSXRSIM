function [num_list] = non_zero_numerologies(numerology, bandwidths)
    
    % returns the numerologies that are to be generated
    num_list = zeros(size(numerology));
    
    for n = 1:length(numerology)
        for freq = 1:size(bandwidths,1)
            if bandwidths(freq,n) ~= 0
                num_list(n) = numerology(n);
            end
        end
    end
end

