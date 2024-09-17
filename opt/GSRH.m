function [S,out] = GSRH(C,X,Omega,reg,verbose)
    [O,T]  = size(X);
    mu0 = reg.mu; %commutativity
    rho = reg.rho; %matrix P
    alpha = reg.alpha;%soft thresholding
    beta = reg.beta; % proximal of P
    t0 = reg.t0;
    max_iters = 10*reg.max_iters;
    Omg = Omega;
    multiples = numel(size(Omega)) > 2;
    delta = reg.delta;%0.001;
    forget = reg.forget;
    
    if verbose
       disp('  -Starting GSR OH optimization') 
    end

    %initialize S as sparse symmetric random matrix
    S = zeros(O);%generate_connected_ER(O,0.1);
    %initialize P as random matrix
    P = zeros(O);

    all_S = zeros(O,O,T);
    all_P = zeros(O,O,T);
    for t = 2:T
          %Update C and gamma
            mu = mu0*t;
            x = X(:,t);
            %
            if forget 
                C = (1-delta)*C + delta*x*x'; % exp3
            else
                C = 1/(t+t0)*((t+t0-1)*C + x*x'); % exp2
            end
            
            lambda = max(eig(C))^2;
            gamma  = 1/(4*mu*lambda);
            
            if multiples
                Omg = Omega(:,:,t);
            end

        for k = 1:max_iters
          %Compute gradient
            R = C*S+P-S*C-P';
            gS = mu*(C*R-R*C);
          %Take gradient descent step
            Ws = S-gamma*gS; 
          %Update S
            S = myS_proximal(Ws,Omg,alpha);
    
          %Compute gradient
            gP = mu*R;
          %Take gradient descent step
            Wp = P-rho*gamma*gP; 
          %Update P
            P = myP_proximal(Wp,beta);
            if sum(sum(abs(P))) < 1e-2
                beta = beta/2;
            end
        end
        if verbose && mod(t,500)==0

            figure(2)
            subplot(221)
            imagesc(S)
            colorbar()
            title('S GSR OH')
            subplot(222)
            imagesc(P)
            colorbar()
            title('P GSR OH')
            subplot(223)
            imagesc(C)
            colorbar()
            title('C')
            sgtitle(num2str(t))
        end
        all_S(:,:,t) = S;
        all_P(:,:,t) = P;
    end

    out.all_S = all_S;
    out.all_P = all_P;

end