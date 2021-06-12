
N1 = 4;             
N2 = 1;
O1 = 4;
O2 = 1; % O2 has only two possibilities : 1 or 4.

n1 = [0 : N1 - 1];
n2 = [0 : N2 - 1];
k1 = [0 : (N1 * O1) - 1];
k2 = [0 : (N2 * O2) - 1];

b1 = zeros(length(n1), length(k1));
b2 = zeros(length(n2), length(k2));

for n1 = 0 : N1 - 1
    for k1 = 0 : (N1 * O1) - 1
        b1(n1+1, k1+1) = exp((1i * 2 * pi * n1 * k1) / (N1 * O1));       
    end
end

for n2 = 0 : N2 - 1
    for k2 = 0 : (N2 * O2) - 1     
        b2(n2+1, k2+1) = exp((1i * 2 * pi * n2 * k2) / (N2 * O2));
    end
end

%normalized ULA precoding vector
X1 = 1 /sqrt(N1) .* b1;
X2 = 1 /sqrt(N2) .* b2;
W2D_rank1_pol1 = kron(X1, X2);

W2D_rank1_pol2 = blkdiag(W2D_rank1_pol1, W2D_rank1_pol1);

%% %%%%%%%%ANTENNA CONFIGURATION%%%%%%%%%%%%%%%%%%%%
% BS and UE Antenna '3gpp-3d' %creating 3gpp directional AE
a_tx = qd_arrayant('3gpp-3d', N1, N2, 3.5e09, 1); 
% a_tx.visualize()
a_tx.coupling = W2D_rank1_pol1(:,13); % or other precoder (like MRT) 
a_tx.combine_pattern;       %combining pattern of M*N to single element
a_tx.visualize()

% % ang_W2D_rank2_pol1 = rad2deg( angle(W2D_rank2_pol1) );
% % 
% % ang_Q1 = rad2deg( angle(Q1) );
% % ang_Q2 = rad2deg( angle(Q2) );
% % plot(ang_Q1)
% % Q = kron(Q1, Q2);

f = gcf();
f.Position = [358.3333 262.3333 891.6667 378.6667];
ax = gca();
ax.View = [90,0]; % YoZ plane (FRONT)
ax.View = [0,90]; % XoY plane (TOP)
ax.View = [0,0]; % XoZ plane  (SIDE)