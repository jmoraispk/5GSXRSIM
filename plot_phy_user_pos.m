for i = 1:16
scatter3(phy_usr_pos(1,i),phy_usr_pos(2,i),phy_usr_pos(3,i))
hold on
end

for i = 1:16    
    disp(i)
    if i == 16
        break
    else
        dist3D(phy_usr_pos(:,i),phy_usr_pos(:,i+1))   
    end
end