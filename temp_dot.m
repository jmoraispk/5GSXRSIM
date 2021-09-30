
N2 = 4;
O2 = 4;
O1 = 4;
gob_col_size = N2 * O2;
q = 4;    
if (4 <= q) && (q <= 7)
    q = q + (gob_col_size - O2) * 1;
    else if (8 <= q) && (q <= 11)
            q = q + (gob_col_size - O2) * 2;
        else if (12 <= q) && (q <= 15)
                q = q + (gob_col_size - O2) * 3;
            end
        end
end
    
%   Step 2: Sum 'offsets' to get the remaining beams in the set
    q_col_idxs = [q + ([0:N1-1] .* (N2*O2*O1))];
    i = 1;
    for q_idx = q_col_idxs
        q_idxs_list(i) = [q_idx + [0:N1-1] .* N1];
        i = i+1;
    end
    q_idxs = q_idxs_list(:);
    
%%
% p1 = 43;
% p2 = 39;
% rot1 = [  0,   4,   8,  12,  64,  68,  72,  76, 128, 132, 136, 140, 192,
%         196, 200, 204]
rot11 =  [ 34,  38,  42,  46,  98, 102, 106, 110, 162, 166, 170, 174, 226, ...
        230, 234, 238];
for p1 = rot11+1    
    for p2 = rot11
        ax1= precoders_matrix(:, p1) ./ norm(precoders_matrix(:, p1));
        ax2= precoders_matrix(:, p2+1) ./ norm(precoders_matrix(:, p2 + 1));
    %     axres(p2+1) = abs(transpose(ax1)*ax2);
        axres(p1, p2 + 1) = dot(ax1, ax2);
        if axres(p1, p2 + 1) > 0.1
            disp(axres(p2 + 1));
            disp(p2 + 1);
        end
    end
end
