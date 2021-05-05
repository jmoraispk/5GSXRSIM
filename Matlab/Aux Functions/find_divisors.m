function [D] = find_divisors(N)
    
    % Finds all divisors of n
    
    % Other more efficient solutions in:
    % https://nl.mathworks.com/matlabcentral/answers/21542-find-divisors-for-a-given-number

    K = 1:ceil(sqrt(N));
    D = K(rem(N,K)==0);
    D = [D sort(N./D)];
end

