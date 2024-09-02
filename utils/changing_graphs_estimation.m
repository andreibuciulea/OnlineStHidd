function out = changing_graphs_estimation(Co, Ctrain, X_test, Omega, reg, Model,verbose)
    T = size(X_test,2);
    if strcmp(Model,'GSR-H')
        % GSR H without online
        [S_hat,~] = GSR_H(Co,reg,verbose);
        S_hat = S_hat/max(max(S_hat));
        allA = S_hat;
    elseif strcmp(Model,'GSR-H-eff')
        % GSR H without online efficient
        [S_hat,~] = GSR_H_eff(Co,Omega,reg,verbose);
        S_hat = S_hat/max(max(S_hat));
        allA = S_hat;
    elseif strcmp(Model,'GSR-O')    
        [~,outO] = GSR_O(Ctrain,X_test,Omega,reg,verbose);
        allA = outO.all_S;
    elseif strcmp(Model,'GSR-OH')
        [~,outOH] = GSR_OH(Ctrain,X_test,Omega,reg,verbose);
        allA = outOH.all_S;
    else
        error('Unknown method')
    end
    out.allShat = allA;
end