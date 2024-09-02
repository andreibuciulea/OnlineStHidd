%Compare several apporaches
    %Hidden
    %online
    %Smoothness+online
    %Online filter identification/expanding graphs
    %Online+hidden (ours)
    %
clear all
%close all
%addpath(genpath('../global_utils'))
addpath('utils\')
addpath('opt\')
%rng(8)
%Define the approaches
Models = {'GSR-O','GSR-OH','GSR-H','GSR-H-eff'};

%Define the parameters
alpha = 1e-6; % soft thresholding for S
beta = 1e-4; % proximal for P 
mu = 1e5; % commutativity
rho = 1e-1; % penalty for P
eps1 = 1e-4;
prms.N = 20;% nodes
O = 15; % observed nodes
M = 1e5;%samples
T = 1e4;%time instants
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
%Generate the graph
    %[A, L] = generate_graph(g_type,prms);
%Generate the signals
    %[~,X,C,~] = generate_graph_signals(sig_type, L, prms, verbose);
%Select hidden nodes
    [s_n, s_h] = select_hidden_nodes(links_type, O, L, C);
%Get observed A,X
Ao = A(s_n,s_n);
Co = C(s_n,s_n);
Xo = X(s_n,:);
X_test = Xo(:,1:T);
X_train = Xo(:,T+1:M);
Ctrain = X_train*X_train'/(M-T);
%Ctrain = Ctrain/max(max(Ctrain));
Ctest = X_test*X_test'/T;
%Compute the estimation for each time instant

reg = struct('alpha',alpha,'beta',beta,'mu',mu,'rho',rho,...
    'max_iters',max_iters,'eps1',eps1);
idx = find(Ao(:,1)==1); 
Omega = zeros(O);Omega(1,idx) = 1;Omega(idx,1) = 1;
%figure(1)
%subplot(223)
%imagesc(Ao)
%colorbar()

%GSR online
[AO_hat,outO] = GSR_O(Ctrain,X_test,Omega,reg,verbose);

%GSR H with online
nAo = norm(Ao,'fro')^2;
[AOH_hat,outOH] = GSR_OH(Ctrain,X_test,Omega,reg,verbose);

% GSR H without online
[S_hat,~] = GSR_H(Co,reg,verbose);
S_hat = S_hat/max(max(S_hat));
err_GSRH = norm(S_hat-Ao,"fro")^2/nAo;
fsc_GSRH = fscore(Ao,mbinarize(S_hat,2));

% GSR H without online efficient
[S_hat,~] = GSR_H_eff(Co,Omega,reg,verbose);
S_hat = S_hat/max(max(S_hat));
err_GSRHef = norm(S_hat-Ao,"fro")^2/nAo;
fsc_GSRHef = fscore(Ao,mbinarize(S_hat,2));

%Compute the estimation error/fscore
allAO = outO.all_S;
allAOH = outOH.all_S;
%allP = out.all_P;

fsc = zeros(T,1);
err = zeros(T,1);
for t = 1:T
   AhO = allAO(:,:,t);
   AhOH = allAOH(:,:,t);
   err(t,1) = norm(AhO-Ao,"fro")^2/nAo;
   fsc(t,1) = fscore(Ao,mbinarize(AhO,2));
   err(t,2) = norm(AhOH-Ao,"fro")^2/nAo;
   fsc(t,2) = fscore(Ao,mbinarize(AhOH,2));
end

figure();
subplot(121)
plot(err,LineWidth=2)
hold on
plot(err_GSRH*ones(T,1))
hold on
plot(err_GSRHef*ones(T,1))
grid on
title('Error')
legend(Models)
subplot(122)
plot(fsc,LineWidth=2)
hold on
plot(fsc_GSRH*ones(T,1))
hold on
plot(fsc_GSRHef*ones(T,1))
grid on
title('Fsc')
legend(Models)

%disp(['Estimation error for GSR-H: ' num2str(err_GSRH)])
%disp(['Fscore for GSR-H: ' num2str(fsc_GSRH)])


%%
alpha = 1e-3; % soft thresholding for S
beta = 1e-3; % proximal for P 
mu = 1e4; % commutativity
rho = 1e-3; % penalty for P
max_iters = 10;
links_type = 'min';
verbose = false;

load('school_real_data.mat','A','YT');
[N,M1,M2] = size(YT);
M = M1*M2;

O = N-10;
T = 10000;

X = reshape(YT,[N,M]);
C = X*X'/M;
%C = C/max(max(C));

A_bin = A>5;
A = A/max(max(A));
L = diag(sum(A))-A;

%Select hidden nodes
[s_n, s_h] = select_hidden_nodes(links_type, O, L, C);
%Get observed A,X
Ao = A(s_n,s_n);nAo = norm(Ao,'fro')^2;
Co = C(s_n,s_n);
Xo = X(s_n,:);
X_test = Xo(:,1:T);
X_train = Xo(:,T+1:M);
Ctrain = X_train*X_train'/(M-T);
%Ctrain = Ctrain/max(max(Ctrain));
Ctest = X_test*X_test'/T;

idx = find(Ao(:,1)==1); 
Omega = zeros(O);Omega(1,idx) = 1;Omega(idx,1) = 1;

% GSR H without online efficient
reg = struct('alpha',alpha,'beta',beta,'mu',mu,'rho',rho,...
    'max_iters',max_iters);
[S_hat,~] = GSR_H_eff(Co,Omega,reg,verbose);
S_hat = S_hat/max(max(S_hat));
err_GSRHef = norm(S_hat-Ao,"fro")^2/nAo;
S_true_bin = mbinarize(S_hat,2);
fsc_GSRHef = fscore(A_bin,S_true_bin);
S_true = S_hat;nSt = norm(S_true,'fro')^2;


%GSR online
[AO_hat,outO] = GSR_O(Ctrain,X_test,Omega,reg,verbose);

%GSR H with online
[AOH_hat,outOH] = GSR_OH(Ctrain,X_test,Omega,reg,verbose);

%Compute the estimation error/fscore
allAO = outO.all_S;
allAOH = outOH.all_S;
%allP = out.all_P;

fsc = zeros(T,1);
err = zeros(T,1);
for t = 1:T
   AhO = allAO(:,:,t);
   AhOH = allAOH(:,:,t);
   err(t,1) = norm(AhO-S_true,"fro")^2/nSt;
   fsc(t,1) = fscore(S_true_bin,mbinarize(AhO,2));
   err(t,2) = norm(AhOH-S_true,"fro")^2/nSt;
   fsc(t,2) = fscore(S_true_bin,mbinarize(AhOH,2));
end

figure();
subplot(121)
plot(err,LineWidth=2)
grid on
title('Error')
legend({'GSR-O','GSR-OH'})
subplot(122)
plot(fsc,LineWidth=2)
grid on
title('Fsc')
legend({'GSR-O','GSR-OH'})