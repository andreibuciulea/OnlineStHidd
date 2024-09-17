%Vary the number of hidden nodes fixing all the other parameters
    %H = 1,2,3,4,5. N= 30, O = 25. 
clear all
addpath('../../utils')
addpath('../../opt')
%rng(4)

%Define the approaches
Models = {'GSR-O','GSR-OH','GSRH'};
nM = numel(Models);
%Define the regularizers 
alpha = 1e-5; % soft thresholding for S
beta = 1e-3; % proximal for P 
mu = 1e2; % commutativity
rho = 1; % penalty for P
delta = 1e-3;
forget = false;

%Define the parameters 
prms.N = 30;% nodes
O = [25,26,27,28,29]; % observed nodes
nO = numel(O);
M = 1e4;%samples
nS = numel(M);
T = M-1;%time instants
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

nG = 60;
all_err = zeros(nG,nO,nM,M-1);
all_fsc = zeros(nG,nO,nM,M-1);

tic
parfor k = 1:nG
    k
    %Generate the graph
    [A, L] = generate_graph(g_type,prms);
    %Generate the signals
    [~,X,C,~] = generate_graph_signals(sig_type, L, prms, verbose);
    err_MO = zeros(nO,nM,M-1);
    fsc_MO = zeros(nO,nM,M-1);
    for o = 1:nO
        Obs = O(o);
        %Select hidden nodes
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
        Ctest = X_test*X_test'/T;
        %Compute the estimation for each time instant
        reg = struct('alpha',alpha,'beta',beta,'mu',mu,'rho',rho,...
            'max_iters',max_iters,'t0',M-T,'delta', delta,'forget', forget);
        err_M = zeros(nM,M-1);
        fsc_M = zeros(nM,M-1);
        for m = 1:nM
            out = graph_estimation(Co, Ctrain, X_test, Omega, reg, Models{m}, Ao, verbose);
            err_M(m,:) = out.err;
            fsc_M(m,:) = out.fsc;
        end
        err_MO(o,:,:) = err_M;
        fsc_MO(o,:,:) = fsc_M;
    end
    all_err(k,:,:,:) = err_MO;
    all_fsc(k,:,:,:) = fsc_MO;
end
toc
%%
load("data_exp1.mat");

linestyle = {'r-','r:','g-','g:','b-','b:'};
figure()
for i = 1:3
    semilogy(squeeze(my_err(4,i,:)),linestyle{1+(i-1)*2},LineWidth=3)
    hold on
    semilogy(squeeze(my_err(1,i,:)),linestyle{2+(i-1)*2},LineWidth=3)
    hold on
end
grid on
xticks([0 5e3 1e4])
xticklabels({'0','5e3','1e4'})
legend('GSR-O H=2','GSR-O H=5','GSR-OH H=2','GSR-OH H=5','Offline H=2','Offline H=5','FontSize',18,'FontWeight','bold','Interpreter','latex')
xlabel('Samples','FontSize',18,'FontWeight','bold','Interpreter','latex')
ylabel('Normalized error','FontSize',18,'FontWeight','bold','Interpreter','latex')
ax = gca;
ax.FontSize = 18;


