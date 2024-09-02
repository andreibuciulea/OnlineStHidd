function [S,out] = GSR_H_eff(C,Omega,reg,verbose)
    O  = size(C,1);
    mu = reg.mu; %commutativity
    rho = reg.rho; %matrix P
    alpha = reg.alpha;%soft thresholding
    beta = reg.beta; % proximal of P
    max_iters = reg.max_iters*100;
    Omega = zeros(size(C));
    
    if verbose
       disp('  -Starting GSR OH optimization') 
    end

    %initialize S as sparse symmetric random matrix
    S = zeros(O);%generate_connected_ER(O,0.1);
    %initialize P as random matrix
    P = randn(O);

    lambda = max(eig(C))^2;
    gamma  = 1/(4*mu*lambda);

    %Update the covariance considering all the previous samples X*X';

    for k = 1:max_iters
      %Compute gradient
        %R = C*S+P-S*C-P';
        %gS = mu*(C*R-R*C);

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
%         figure(1)
%         subplot(221)
%         imagesc(S)
%         colorbar()
%         title('S GSR OH')
%         subplot(222)
%         imagesc(P)
%         colorbar()
%         title('P GSR OH')
    end
    %if verbose 


    %end
    %all_S(:,:) = S;
    %all_P(:,:) = P;

    out.all_S = S;
    out.all_P = P;

end






