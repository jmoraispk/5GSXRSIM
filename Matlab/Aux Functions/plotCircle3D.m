function plotCircle3D(center,normal,radius, C)


    % From https://www.mathworks.com/matlabcentral/fileexchange/26588-plot-circle-in-3d
    % I Didn't do any modifications to Christian Reinbacher work, besides
    % the last 2 lines

    theta=0:0.01:2*pi;
    v=null(normal);
    points=repmat(center',1, ...
               size(theta,2))+radius*(v(:,1)*cos(theta)+v(:,2)*sin(theta));
    plot3(points(1,:),points(2,:),points(3,:),'r-');
    
    
    % can be a RGB tripled like [1 1 0] (yellow) or [0.5 0.25 0](chocolate)
    if ~isempty(C)
        fill3(points(1,:), points(2,:),points(3,:), C);
    end
end