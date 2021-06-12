function [list_of_angles] = subarray_square_panel_pattern(n, a)
    

    % TODO : make this general on square arrays with side n!

    % Given a square array of side (side), beamform into 4 different angles
    % [+-a, 0] and [0, +-a], and apply these angles to the patterns
    
    % Returns a cell array with the correct dimensions
    
    p1 = [-a, 0];
    p2 = [0, -a];
    p3 = [0, a];
    p4 = [a, 0];
    
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
    % [p3 p3 p1 p1        LOOK HERE TO UNDERSTAND THE ORDER OF ANGLES ABOVE
    %  p3 p3 p1 p1        (Still from the back! note that if a user is at
    %  p4 p4 p2 p2         the right of the antenna, p1 will cover that 
    %  p4 p4 p2 p2]        region since -30 would point to the right!
    
    % How fun! You have to put the '...' or else it considers a line change
    list_of_angles = {p1 p1 p3 p3 ...
                      p1 p1 p3 p3 ...
                      p2 p2 p4 p4 ...
                      p2 p2 p4 p4};
end