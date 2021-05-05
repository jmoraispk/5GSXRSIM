function [points] = uniform(a,b, siz)
    %UNIFORM Samples a uniform distribution
    %   Detailed explanation goes here
    a2 = a/2;
    b2 = b/2;
    mu = a2+b2;
    sig = b2-a2;
    points = mu + sig .* (2*rand(siz)-1);
end

