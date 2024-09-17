function [S_hat,out] = GSR_H_constr(C,Omega,reg,verbose)
N = size(C,1);
alpha = reg.alpha;
rho = reg.rho;
mu = reg.mu;
%beta = 1;
mu = 1e8;
alpha = 1;
rho = 1;
epsilon = 1e-8;

if verbose
   disp('  -Starting GSR Low Rank optimization...') 
end

cvx_begin quiet
    variable S_hat(N,N) symmetric
    variable P_hat(N,N) 
    %        Sparse matrix  +  low rank matrix
    minimize (alpha*norm(S_hat(:),1) + rho*sum(norms(P_hat)))
    subject to
        norm(C*S_hat + P_hat - S_hat*C - P_hat', 'fro') <= epsilon;
        diag(S_hat) <= 1e-6;
        S_hat >= 0;
        S_hat*ones(N,1) >= 1;
        S_hat(Omega~=0) >= Omega(Omega~=0); 
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






