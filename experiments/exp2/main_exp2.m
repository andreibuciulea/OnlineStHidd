%Vary the number of hidden nodes fixing all the other parameters
    %H = 1,2,3,4,5. N= 30, O = 25. 
    % M = 1e4 T = 9e3

clear all
%close all
%addpath(genpath('../global_utils'))
addpath('utils')
addpath('opt')
rng(1)

%Define the approaches
Models = {'GSR-O','GSR-OH','GSR-H','GSR-H-eff'};
nM = numel(Models);
%Define the regularizers 
alpha = 1e-4; % soft thresholding for S
beta = 1e-2; % proximal for P 
mu = 1; % commutativity
rho = 1e-3; % penalty for P

%Define the parameters 
prms.N = 30;% nodes
O = [25,26,27,28,29]; % observed nodes
nO = numel(O);
M = 1e4;%samples
nS = numel(M);
T = 9e3;%time instants
max_iters = 10;
prms.M = M(end); 
prms.p = 0.1; % connection probability
g_type = 'ER';
links_type = 'min';
sig_type = 'ST';
prms.norm_L = false;
prms.sigma = 0;
prms.sampled = true;
verbose = false;
%Generate the graph
    [A, L] = generate_graph(g_type,prms);
%Generate the signals
    [~,X,C,~] = generate_graph_signals(sig_type, L, prms, verbose);
%Select hidden nodes

for k = 1:5
for o = 1: nO 
    Obs = O(o);
    [s_n, s_h] = select_hidden_nodes(links_type, Obs, L, C);
    %Get observed A,C,X
    Ao = A(s_n,s_n);
    idx = find(Ao(:,1)==1); 
    Omega = zeros(Obs);Omega(1,idx) = 1;Omega(idx,1) = 1;
    Co = C(s_n,s_n);
    Xo = X(s_n,:);

    X_test = Xo(:,1:T);
    X_train = Xo(:,T+1:M);
    Ctrain = X_train*X_train'/(M-T);
    %Ctrain = Ctrain/max(max(Ctrain));
    Ctest = X_test*X_test'/T;
    %Compute the estimation for each time instant
    reg = struct('alpha',alpha,'beta',beta,'mu',mu,'rho',rho,...
        'max_iters',max_iters,'t0',M-T);
    for m = 1:nM
        out = graph_estimation(Co, Ctrain, X_test, Omega, reg, Models{m}, Ao, verbose);
        est_err{o,m} = out.err;
        est_fsc{o,m} = out.fsc;
    end
end

%
figure()
for j = 1:5
    subplot(2,3,j)
    for i = 1:4
        plot(est_err{j,i},Linewidth=2)
        hold on
    end
    title(['Observed nodes = ' num2str(O(j)) ' of 30'])
    legend(Models)
    grid on
    xlabel('Samples')
end
filename = ['figure' num2str(k) '.fig'];
savefig(filename);
end