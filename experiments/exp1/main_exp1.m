%Vary the number of samples fixing all the other parameters. 
    % M = 1e2,1e3,1e4,1e5
    % T = 9e1,9e2,9e3,9e4

clear all
%close all
addpath(genpath('../../../global_utils'))
addpath('../../utils')
addpath('../../opt')
%rng(1)

%Define the approaches
Models = {'GSR-O','GSR-OH','GSR-H','GSR-H-eff'};
nM = numel(Models);
%Define the regularizers 
alpha = 1e-4; % soft thresholding for S
beta = 1e-3; % proximal for P 
mu = 1; % commutativity
rho = 1e-2; % penalty for P

%Define the parameters 
prms.N = 30;% nodes
O = 25; % observed nodes
M = [1e2,1e3,1e4,1e5];%samples
nS = numel(M);
T = [9e1,9e2,9e3,9e4];%time instants
max_iters = 10;
prms.M = M(end); 
prms.p = 0.1; % connection probability
g_type = 'ER';
links_type = 'rand';
sig_type = 'ST';
prms.norm_L = false;
prms.sigma = 0;
prms.sampled = true;
verbose = false;
for k = 1:5 
    %Generate the graph
        [A, L] = generate_graph(g_type,prms);
    %Generate the signals
        [~,X,C,~] = generate_graph_signals(sig_type, L, prms, verbose);
    %Select hidden nodes
        [s_n, s_h] = select_hidden_nodes(links_type, O, L, C);
    %Get observed A,C,X
    Ao = A(s_n,s_n);
    idx = find(Ao(:,1)==1); 
    Omega = zeros(O);Omega(1,idx) = 1;Omega(idx,1) = 1;
    Co = C(s_n,s_n);
    Xo = X(s_n,:);
    
    
    for n = 1:nS
        Mn = M(n);
        Tn = T(n);
        X_test = Xo(:,1:Tn);
        X_train = Xo(:,Tn+1:Mn);
        Ctrain = X_train*X_train'/(Mn-Tn);
        %Ctrain = Ctrain/max(max(Ctrain));
        Ctest = X_test*X_test'/Tn;
        Com = Xo(:,1:Mn)*Xo(:,1:Mn)'/Mn;
        %Compute the estimation for each time instant
        reg = struct('alpha',alpha,'beta',beta,'mu',mu,'rho',rho,...
            'max_iters',max_iters,'t0',Mn-Tn);
        for m = 1:nM
            out = graph_estimation(Com, Ctrain, X_test, Omega, reg, Models{m}, Ao, verbose);
            est_err{n,m} = out.err;
            est_fsc{n,m} = out.fsc;
        end
    end
    
    %
    figure()
    for j = 1:4
        subplot(2,2,j)
        for i = 1:4
            plot(est_err{j,i},Linewidth=2)
            hold on
        end
        title(['M = ' num2str(M(j)) ' and  T = ' num2str(T(j))])
        legend(Models)
        grid on
        xlabel('Samples')
    end
    filename = ['figure' num2str(k) '.fig']; 
    savefig(filename);
end