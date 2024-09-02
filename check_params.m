clear all
%close all
addpath('utils');
addpath('opt');

P1 = logspace(-4,-4,1); %alpha
P2 = logspace(-2,-2,1); %beta
P3 = logspace(0,0,1); %mu
P4 = logspace(-3,-3,1); %rho
P5 = logspace(-2,-2,1); %eps1


%generate the grid serch params
n_p1 = numel(P1);n_p2 = numel(P2);n_p3 = numel(P3);n_p4 = numel(P4);n_p5 = numel(P5);
n_tot = n_p1*n_p2*n_p3*n_p4*n_p5;
a_p1 = cell(1,n_tot);a_p2 = a_p1;a_p3 = a_p1;a_p4 = a_p1;a_p5 = a_p1;
i = 1;
for p1 = 1:n_p1
    for p2 = 1:n_p2
        for p3 = 1:n_p3
            for p4 = 1:n_p4
                for p5 = 1:n_p5
                    a_p1{i} = P1(p1);
                    a_p2{i} = P2(p2);
                    a_p3{i} = P3(p3);
                    a_p4{i} = P4(p4);
                    a_p5{i} = P5(p5);
                    i = i+1;
                end
            end
        end
    end
end


%edit params acording to your necesity
nG = 30; %number of graphs
N = 30;  % nodes
O = 25;  % observed nodes
M = 1e4; % samples 
T = 1e3; % time instants
max_iters = 10;
g_type = 'ER';
links_type = 'min';
sig_type = 'ST';
p = 0.1; % connection probability
norm_L = false;
sigma = 0;
sampled = true;
verbose = false;


alg_prms = struct('alpha',a_p1,'beta',a_p2,'mu', a_p3,'rho',a_p4,...
                    'eps1',a_p5,'t0',T,'max_iters',max_iters);

prms = struct('N',N,'M',M,'p',p,'norm_L',norm_L,'sigma',sigma,'sampled',sampled);

res = zeros(nG,n_tot,T,2);
parfor g = 1:nG
    res_k = zeros(n_tot,T,2);
    %Generate the graph
    [A, L] = generate_graph(g_type,prms);
    %Generate the signals
    [~,X,C,~] = generate_graph_signals(sig_type, L, prms, verbose);
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
    idx = find(Ao(:,1)==1); 
    Omega = zeros(O);Omega(1,idx) = 1;Omega(idx,1) = 1;

    nAo = norm(A,'fro')^2;
    for k = 1:n_tot
        [AOH_hat,outOH] = GSR_OH(Ctrain,X_test,Omega,alg_prms(k),verbose);
        res_k(k,:,:) = compute_performance(outOH,T,Ao);
        %[S_hat,~] = GSR_H_eff(Co,Omega,alg_prms(k),verbose);
        %S_hat = S_hat/max(max(S_hat));
        %res_k(k,1,1) = norm(S_hat-Ao,"fro")^2/nAo;
        %res_k(k,1,2) = fscore(Ao,mbinarize(S_hat,2));
    end
    res(g,:,:,:) = res_k;
end

%figure()
%plot(squeeze(median(res(:,:,1,1))))

%
v = 1:n_tot;
results = squeeze(median(res(:,:,:,1)))';
cellArray = num2cell(v); 
%lgd = cellfun(@num2str, cellArray, 'UniformOutput', false); 
figure()
subplot(121)
plot(results)
%legend(lgd)
[val,idx] = min(results(end,:));
title(['Error: Value:' num2str(val) ', Idx:' num2str(idx)])
subplot(122)
results = squeeze(median(res(:,:,:,2)))';
plot(results)
%legend(lgd)
[val,idx] = max(results(end,:));
title(['Fscore: Value:' num2str(val) ', Idx:' num2str(idx)])



%%

pidx = [1,34];
figure()
subplot(121)
results = squeeze(median(res(:,:,:,1)))';
plot(results(:,pidx))
%legend(lgd)
[val,idx] = min(results(end,:));
title(['Error: Value:' num2str(val) ', Idx:' num2str(idx)])
subplot(122)
results = squeeze(median(res(:,:,:,2)))';
plot(results(:,pidx))
%legend(lgd)
[val,idx] = max(results(end,:));
title(['Fscore: Value:' num2str(val) ', Idx:' num2str(idx)])
