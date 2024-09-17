function [S,out] = GSR_H_eff(C,Omega,reg,verbose)
    O  = size(C,1);
    mu = reg.mu; %commutativity
    rho = reg.rho; %matrix P
    alpha = reg.alpha;%soft thresholding
    beta = reg.beta; % proximal of P
    max_iters = reg.max_iters;
    max_iters = 1e5;
    
    if verbose
       disp('  -Starting GSR OH optimization') 
    end

    %initialize S as sparse symmetric random matrix
    S = zeros(O);%generate_connected_ER(O,0.1);
    %initialize P as random matrix
    P = randn(O);

    lambda = max(eig(C))^2;
    gamma  = 1/(4*mu*lambda);

    for k = 1:max_iters
      %Compute gradient
        R = S*C+P-C*S-P';
        gS = mu*(R*C-C*R);
      %Take gradient descent step
        Ws = S-gamma*gS; 
      %Update S
        S = myS_proximal(Ws,Omega,alpha);

      %Compute gradient
        gP = 2*mu*R;
      %Take gradient descent step
        Wp = P-rho*gamma*gP; 
      %Update P
        P = myP_proximal(Wp,beta);
    end

    out.all_S = S;
    out.all_P = P;

end






