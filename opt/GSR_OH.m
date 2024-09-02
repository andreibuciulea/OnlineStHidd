function [S,out] = GSR_OH(C,X,Omega,reg,verbose)
    [O,T]  = size(X);
    mu = 100*reg.mu; %commutativity
    rho = reg.rho; %matrix P
    alpha = reg.alpha;%soft thresholding
    beta = reg.beta; % proximal of P
    t0 = reg.t0;
    max_iters = reg.max_iters;
    Omg = Omega;
    multiples = numel(size(Omega)) > 2;
    
    if verbose
       disp('  -Starting GSR OH optimization') 
    end

    %initialize S as sparse symmetric random matrix
    S = zeros(O);%generate_connected_ER(O,0.1);
    %initialize P as random matrix
    P = randn(O);

    all_S = zeros(O,O,T);
    all_P = zeros(O,O,T);
    for t = 2:T
          %Update C and gamma
            x = X(:,t);
            C = 1/(t+t0)*((t+t0-1)*C + x*x'); % t0 samples in the train
            %covariance
            %C = 1/t*((t-1)*C + x*x');
            lambda = max(eig(C))^2;
            gamma  = 1/(4*mu*lambda);
            
            if multiples
                Omg = Omega(t);
            end

        for k = 1:max_iters
          %Compute gradient
            %R = C*S+P-S*C-P';
            %gS = mu*(C*R-R*C);

            R = S*C+P-C*S-P';
            gS = mu*(R*C-C*R);
          %Take gradient descent step
            Ws = S-gamma*gS; 
          %Update S
            S = myS_proximal(Ws,Omg,alpha);
    
          %Compute gradient
            gP = 2*mu*R;
          %Take gradient descent step
            Wp = P-rho*gamma*gP; 
          %Update P
            P = myP_proximal(Wp,beta);
        end
        if verbose 
        figure(1)
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






