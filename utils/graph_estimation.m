function out = graph_estimation(Co, Ctrain, X_test, Omega, reg, Model,Ao,verbose)
    T = size(X_test,2);
    nAo = norm(Ao,'fro')^2;
    err = zeros(T,1);fsc = zeros(T,1);
    if strcmp(Model,'GSR-H')
        % GSR H without online
        [S_hat,~] = GSR_H(Co,reg,verbose);
        S_hat = S_hat/max(max(S_hat));
        err = norm(S_hat-Ao,"fro")^2/nAo*ones(T,1);
        fsc = fscore(Ao,mbinarize(S_hat,2))*ones(T,1);
        allA = S_hat;
    elseif strcmp(Model,'GSR-H-eff')
        % GSR H without online efficient
        [S_hat,~] = GSR_H_eff(Co,Omega,reg,verbose);
        S_hat = S_hat/max(max(S_hat));
        err = norm(S_hat-Ao,"fro")^2/nAo*ones(T,1);
        fsc = fscore(Ao,mbinarize(S_hat,2))*ones(T,1);
        allA = S_hat;
    elseif strcmp(Model,'GSR-O')    
        [~,outO] = GSR_O(Ctrain,X_test,Omega,reg,verbose);
        allA = outO.all_S;
        for t = 1:T
           AhO = allA(:,:,t);
           err(t) = norm(AhO-Ao,"fro")^2/nAo;
           fsc(t) = fscore(Ao,mbinarize(AhO,2));
        end
    elseif strcmp(Model,'GSR-OH')
        [~,outOH] = GSR_OH(Ctrain,X_test,Omega,reg,verbose);
        allA = outOH.all_S;
        for t = 1:T
           AhOH = allA(:,:,t);
           err(t) = norm(AhOH-Ao,"fro")^2/nAo;
           fsc(t) = fscore(Ao,mbinarize(AhOH,2));
        end
    else
        error('Unknown method')
    end
    out.err = err;
    out.fsc = fsc;
    out.allShat = allA;
end