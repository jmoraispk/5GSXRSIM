function [ang] = calc_2Dangle_from_pos(pos1,pos2)
    %CALC_ANGLE_FROM_POS From 2 (x,y) positions, return the angle between them
    %pos 1 is the origin of the angle!
    
    %normally, we need to calculate the tangent of the slope (atan(dy/dx)),
    %and then based on the values of dy and dx, make a decision on the
    %sign.  atan2 does the atan and this decision   
    
    
    %calc the slope of the line segment that connects them
    dy = (pos2(2) - pos1(2));
    dx = (pos2(1) - pos1(1));
    
    %compute angle with decision on the sign include
    ang = atan2(dy,dx);
    
end

