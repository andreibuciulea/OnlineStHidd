%Generate signlas associated with two graphs: 
    % One with two communities
    % The other one with four communities. 
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
beta = 1e-1; % proximal for P 
mu = 1; % commutativity
rho = 1e-3; % penalty for P

%Define the parameters 
N = 30;
prms.N = N;% nodes
O = 28;%observed nodes
M = 1e4;%samples
nS = numel(M);
T = 9e3;%time instants
max_iters = 20;
prms.M = M(end); 
links_type = 'min';
sig_type = 'ST';
prms.norm_L = false;
prms.sigma = 0;
prms.sampled = true;
verbose = false;


%Generate two communities graph 
    %p: intra cluster prob, q: inter cluster prob, k = 2 number of clusters
    sbm_prms = struct('p', 0.2,'q',0.01);
    G = gsp_stochastic_block_graph(N,2,sbm_prms);
    A = full(G.A); L = diag(sum(A))-A;
%Generate the signals
    [~,X,C,~] = generate_graph_signals(sig_type, L, prms, verbose);
%Select hidden nodes
    [s_n, s_h] = select_hidden_nodes(links_type, O, L, C);
%Get observed A,C,X
    Ao1 = double(A(s_n,s_n));nAo1 = norm(Ao1,'fro')^2;
    idx = find(Ao1(:,1)==1); 
    Omega1 = zeros(O);Omega1(1,idx) = 1;Omega1(idx,1) = 1;
    Co1 = C(s_n,s_n);
    Xo1 = X(s_n,:);
    X_test1 = Xo1(:,1:T);
    X_train1 = Xo1(:,T+1:M);
    Ctrain1 = X_train1*X_train1'/(M-T);

%Generate four communities graph 
    %p: intra cluster prob, q: inter cluster prob, k = 4 number of clusters
    sbm_prms = struct('p', 0.5,'q',0.1);
    G = gsp_stochastic_block_graph(N,4,sbm_prms);
    A = full(G.A); L = diag(sum(A))-A;
%Generate the signals
    [~,X,C,~] = generate_graph_signals(sig_type, L, prms, verbose);
%Select hidden nodes
    [s_n, s_h] = select_hidden_nodes(links_type, O, L, C);
%Get observed A,C,X
    Ao2 = double(A(s_n,s_n));nAo2 = norm(Ao2,'fro')^2;
    idx = find(Ao2(:,1)==1); 
    Omega2 = zeros(O);Omega2(1,idx) = 1;Omega2(idx,1) = 1;
    Co2 = C(s_n,s_n);
    Xo2 = X(s_n,:);
    X_test2 = Xo2(:,1:T);

    X_test12 = [X_test1 X_test2];
    Xo12 = [Xo1 Xo2];
    Co12 = Xo12*Xo12'/(2*T);
    Omegas = [repmat(Omega1,1,T) repmat(Omega2,1,T)];
    Omegas = reshape(Omegas, [O,O,2*T]);

    %Compute the estimation for each time instant
    reg = struct('alpha',alpha,'beta',beta,'mu',mu,'rho',rho,...
        'max_iters',max_iters,'t0',M-T);
    err = zeros(nM,2*T);
    fsc = zeros(nM,2*T);
    for m = 1:nM
        out = changing_graphs_estimation(Co12, Ctrain1, X_test12, Omegas, reg, Models{m}, verbose);
        allShat = out.allShat;

        if strcmp(Models{m},'GSR-H') || strcmp(Models{m},'GSR-H-eff')
            err1 = norm(allShat-Ao1,"fro")^2/nAo1;
            fsc1 = fscore(Ao1,mbinarize(allShat,2));
            err2 = norm(allShat-Ao2,"fro")^2/nAo2;
            fsc2 = fscore(Ao2,mbinarize(allShat,2));
            err(m,:) = [err1*ones(1,T) err2*ones(1,T)]; 
            fsc(m,:) = [fsc1*ones(1,T) fsc2*ones(1,T)];
        end
    
        if strcmp(Models{m},'GSR-O') || strcmp(Models{m},'GSR-OH')
            for t = 1:T
                Shat = allShat(:,:,t);
                err(m,t) = norm(Shat-Ao1,"fro")^2/nAo1;
                fsc(m,t) = fscore(Ao1,mbinarize(Shat,2));
            end
            
            for t = T+1:2*T
                Shat = allShat(:,:,t);
                err(m,t) = norm(Shat-Ao2,"fro")^2/nAo2;
                fsc(m,t) = fscore(Ao2,mbinarize(Shat,2));
            end
        end
    end

%
figure()
subplot(121)
plot(err',LineWidth=2)
xlabel('Samples')
ylabel('Error')
legend(Models)
grid on
subplot(122)
plot(fsc',LineWidth=2)
grid on
xlabel('Samples')
ylabel('Fscore')
legend(Models)
