function [S,out] = GSR_O(C,X,Omega,reg,verbose)
    [O,T]  = size(X);
    mu = reg.mu; %commutativity
    alpha = reg.alpha;%soft thresholding
    max_iters = reg.max_iters;
    t0 = reg.t0;
    Omg = Omega;
    multiples = numel(size(Omega)) > 2;
    if verbose
       disp('  -Starting GSR Online optimization') 
    end

    %initialize S as sparse symmetric random matrix
    S = zeros(O);%generate_connected_ER(O,0.1);

    all_S = zeros(O,O,T);
    for t = 2:T
      %Update C and gamma
        x = X(:,t);
        %Cx = x*x';
        %Cx = Cx/max(max(Cx));
        %C = 1/t*((t-1)*C + Cx);
        %C = C/max(max(C));
        C = 1/(t+t0)*((t+t0-1)*C + x*x'); % t0 samples in the train
        %C = 1/t*((t-1)*C + x*x');
        lambda = max(eig(C))^2;
        gamma  = 1/(4*mu*lambda);
        if multiples
            Omg = Omega(:,:,t);
        end

        for k = 1:max_iters
          %Compute gradient
            R = S*C-C*S;
            gS = mu*(R*C-C*R);
          %Take gradient descent step
            Ws = S-gamma*gS; 
          %Update S
            S = myS_proximal(Ws,Omg,alpha);
        end
        %if verbose 
        % figure(1)
        % subplot(221)
        % imagesc(S)
        % colorbar()
        % title('S GSR O')

        %end
        all_S(:,:,t) = S;
    end
    out.all_S = all_S;

end






