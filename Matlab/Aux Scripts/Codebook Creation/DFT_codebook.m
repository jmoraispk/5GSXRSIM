function [W] = DFT_codebook(N1, N2, O1, O2, pol, RI, cophase_fact)
    N1 = 4;             
    N2 = 4;
    O1 = 4; % O1 is always 4.
    O2 = 4; % O2 is always 1 or 4.
    % pol = 3;
    % RI = 1;

%     cophase_fact = [1, -1, 1i, -1i];
    k = N1 * N2 * O1 * O2;
    n1 = [0 : N1 - 1];
    n2 = [0 : N2 - 1];
    k1 = [0 : (N1 * O1) - 1];
    k2 = [0 : (N2 * O2) - 1];

    b1 = zeros(length(n1), length(k1));
    b2 = zeros(length(n2), length(k2));

    % Horizontal dimension DFT vector (ULA).

    for n1 = 0 : N1 - 1
        for k1 = 0 : (N1 * O1) - 1
            b1(n1+1, k1+1) = exp(-(1i * 2 * pi * n1 * k1) / (N1 * O1));       
        end
    end

    % Vertical dimension DFT vector (ULA).

    for n2 = 0 : N2 - 1
        for k2 = 0 : (N2 * O2) - 1     
            b2(n2+1, k2+1) = exp(-(1i * 2 * pi * n2 * k2) / (N2 * O2));
        end
    end

    % Normalized horizontal and vertical dimension DFT vectors.

    X1 = 1 /sqrt(N1) .* b1;
    X2 = 1 /sqrt(N2) .* b2;

    % Codebook for ULA or URA antenna array for single polarisation and rank.
    % This codebook is of size N1.N2 x N1.N2.O1.O2. There are total N1.N2.O1.O2
    % precoders (columns) in this codebook. All of the columns are not
    % orthogonal to each other. There are N1.N2 orthogonal beams and there can
    % be O1.O2 such sets of orthogonal beams.

    W2D_rank1_pol1 = kron(X1, X2);
    W2D_block = blkdiag(W2D_rank1_pol1, W2D_rank1_pol1);

    if (pol == 1) && (RI == 1) 
        W = W2D_rank1_pol1;
        
    else
        if (pol == 1) && (RI == 2)
             W = W2D_block;

        else
            if (pol == 3) && (RI == 1)
                % This creates same precoder (column) for both polarisations. 
                % Total number of precoders remain same as pol 1 (N1.N2.O1.O2).
                % W = [2 * N1 * N2, k]
                order_precod = ones(size(N1*N2));
                m = 1;
                for p = 1 : N1
                    for q = 0 : N2 - 1
                        order_precod(m) = p + q * N1;
                        m = m + 1;
                    end
                end
                W3 = zeros(size(W2D_rank1_pol1));
                for j = 1 : (N1 * N2)
                    W3(j, :) = W2D_rank1_pol1(order_precod(j), :);
                end
                W = [W3 ; cophase_fact .* W3];
                
                
                % Add the block diagram multiplication method here after
                % multiplying correct sized ph with W2D_block and then compare 
                % of both yields the same result.
                % W2 = W2D_block * ph;

            else
                if (pol == 3) && (RI == 2)
                    % For rank 1 or first layer of transmission, 1 precoder is
                    % selected for both polarizations from the first 256, for next
                    % rank, another precoder is selected from the next 256 columns.
                    %W = [2 * N1 * N2, 2 * k] 
                    % CHECK THIS FIRST QUADRANT! the rest is prob right...
                    first_quadrant = ...
                        W2D_block(1 : (N1 * N2), 1 : (k/2));
                    
                    % 1 : (N1 * N2) on rows means 1st pol
                    % (N1 * N2) + 1 : 2 * N1 * N2 on rows means 2nd pol
                    % 1 : k/2 on columns means 1st rank
                    % k/2 + 1 : k on columns means 1st rank
                    
                    % First rank, 2nd pol
                    W((N1 * N2) + 1 : 2 * N1 * N2, 1 : (k/2) ) = ...
                        cophase_fact * first_quadrant;
                    % Second rank, 1st pol
                    W(1:(N1 * N2) + 1, k/2 +1 : k) = first_quadrant;
                    % Second rank, 2nd pol
                    W((N1 * N2) + 1 : 2 * N1 * N2, k/2 +1 : k) = ...
                        -cophase_fact * first_quadrant;

                end           
            end
        end 
    end
end