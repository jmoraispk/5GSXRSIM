function [val] = check_size(A, sizee)
    %CHECK_SIZE returns 1 if the size of A is 'size'
    
    val = isequal(size(A),sizee);
end

