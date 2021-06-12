function [list_of_angles] = subarray_flower_pattern(n, a)
    

    % TODO : make this general on square arrays with side n!

    % Given a square array of side (side), beamform into 4 different angles
    % Sides beamform to the side, 
    % centre to the centre, and diagonals in diagonal!
    
    % Centre
    pc0 = [0,0];
    % Sides
    ps1 = [-a, 0];
    ps2 = [0 , a];
    ps3 = [a, 0];
    ps4 = [0, -a];
    % Diagonals
    pd1 = [-a, a];
    pd2 = [a, a];
    pd3 = [a, -a];
    pd4 = [-a, -a];
    
    % HAVE THE FOLLOWING IN MIND WHEN PICKING ANGLES!
    % From the front, an array has the indices like: 
    % [1 5  9 13
    %  2 6 10 14
    %  3 7 11 15  
    %  4 8 12 16]
    % From the back, the indices would be like this:
    % [13  9  5 1
    %  14 10  6 2
    %  15 11  7 3  
    %  16 12  8 4]
    % and below are how the angles relate with the indices of the subarrays
    %      | 
    %      V        
    % [pd2 ps2 ps2 pd1        LOOK HERE TO UNDERSTAND THE ORDER OF ANGLES ABOVE
    %  ps3 pc0 pc0 ps1        (Still from the back! note that if a user is at
    %  ps3 pc0 pc0 ps1         the right of the antenna, p1 will cover that 
    %  pd3 ps4 ps4 pd4]        region since -30 would point to the right!
    
    % and then we invert to give the proper order of the indices expected!
    list_of_angles = {pd1 ps2 ps2 pd2 ...
                      ps1 pc0 pc0 ps3 ...
                      ps1 pc0 pc0 ps3 ...
                      pd4 ps4 ps4 pd3};
end