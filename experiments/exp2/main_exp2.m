%Compare several apporaches
clear all
addpath(genpath('../../../global_utils'))
addpath('../../utils')
addpath('../../opt')


Models = {'GSR-O','GSR-O','GSR-O','GSR-OH','GSR-OH','GSR-OH','GSR-OH'};
Iterations = [1,10,100,1,10,100,200];

nM = numel(Models);

nG = 60;
%Define the parameters
alpha = 1e-5; % soft thresholding for S
beta = 1e-3; % proximal for P 
mu = 1e2; % commutativity
rho = 1; % penalty for P
delta = 1e-3;
forget = false;

prms.N = 30;% nodes
O = 28; % observed nodes

M = 5e4;%samples
T = M-1;%time instants
max_iters = 10;
prms.M = M; 
prms.p = 0.1; % connection probability
g_type = 'ER';
links_type = 'rand';
sig_type = 'ST';
prms.norm_L = false;
prms.sigma = 0;
prms.sampled = true;
verbose = false;
all_err = zeros(nG,nM,M-1);
all_fsc = zeros(nG,nM,M-1);
tic
parfor g = 1:nG
    g
    %Generate the graph
        [A, L] = generate_graph(g_type,prms);
    %Generate the signals
        [~,X,C,~] = generate_graph_signals(sig_type, L, prms, verbose);
    %Select hidden nodes
        [s_n, s_h] = select_hidden_nodes(links_type, O, L, C);
    %Get observed A,X
    Ao = A(s_n,s_n);nAo = norm(Ao,'fro')^2;
    Co = C(s_n,s_n);
    Xo = X(s_n,:);
    X_test = Xo(:,1:T);
    X_train = Xo(:,T+1:M);
    Ctrain = X_train*X_train'/(M-T);
    Ctest = X_test*X_test'/T;
    %Compute the estimation for each time instant
    
    reg = struct('alpha',alpha,'beta',beta,'mu',mu,'rho',rho,...
            'max_iters',max_iters,'t0',M-T,'delta',delta,'forget',forget);
    
    idx = find(Ao(:,1)==1); 
    Omega = zeros(O);Omega(1,idx) = 1;Omega(idx,1) = 1;
    
    est_err = zeros(nM,M-1);
    est_fsc = zeros(nM,M-1);
    for m = 1: nM
        m
        reg.max_iters = Iterations(m);
        out = graph_estimation(Co, Ctrain, X_test, Omega, reg, Models{m}, Ao, verbose);
        est_err(m,:) = out.err';
        est_fsc(m,:) = out.fsc';
    end
    all_err(g,:,:) = est_err;
    all_fsc(g,:,:) = est_fsc;
end
toc
%%
load("data_exp2.mat");
Models = {'GSR-O','GSR-O','GSR-O','GSR-OH','GSR-OH','GSR-OH','Offline'};
%[a,b] = sort(all_err(:,7,end));
%my_err = squeeze(mean(all_err(b(1:20),:,:)));
my_x = 1:M-1;
lgd = cell(nM,1);
step = 200;
figure()
for i = 1:nM
    x = my_x(1:step:end);
    y = my_err(i,1:step:end);
    loglog(x,y,Linewidth=3)
    hold on
    lgd{i} = [Models{i} '-' num2str(Iterations(i))];
end
xlim([1e2 5e4])
xticks([1e2 3e2 1e3 3e3 1e4 5e4])
xticklabels({'1e2','3e2','1e3','3e3','1e4','5e4'})
%ylim([0.18 0.45])
%yticks([0.18 0.25 0.35 0.45])
%yticklabels({'0.18','0.25','0.35','0.45'})
ax = gca;
ax.FontSize = 18;
hlg = legend(lgd,'FontSize',18,'FontWeight','bold','Interpreter','latex');
hlg.Color = 'none';
grid on
xlabel('Samples','FontSize',18,'FontWeight','bold','Interpreter','latex')
ylabel('Normalized error','FontSize',18,'FontWeight','bold','Interpreter','latex')




