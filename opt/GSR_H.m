function [S_hat,out] = GSR_H(C,reg,verbose)
N = size(C,1);
alpha = reg.alpha;
beta = reg.beta;
mu = reg.mu;
alpha = 4e-3;
beta = 1;
mu = 1e8;

if verbose
   disp('  -Starting GSR Low Rank optimization...') 
end

cvx_begin quiet
    variable S_hat(N,N) symmetric
    variable P_hat(N,N) 
    %        Sparse matrix  +  low rank matrix
    %square the frobenius 
    minimize (norm(S_hat(:),1) + beta*sum(norms(P_hat)) +...
              mu*norm(C*S_hat + P_hat - S_hat*C - P_hat', 'fro'))
    subject to
    diag(S_hat) <= 1e-6;
    S_hat >= 0;
    S_hat*ones(N,1) >= 1;

cvx_end
out.P = P_hat;
if verbose 
    figure(1)
    subplot(1,2,1)
    imagesc(S_hat)
    colorbar()
    title('S GSt')
    subplot(1,2,2)
    imagesc(P_hat)
    colorbar()
    title('K GSt')
end
end






