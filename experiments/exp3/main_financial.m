clear all;
alpha = 1e-3; % soft thresholding for S
beta = 1e3; % proximal for P 
mu = 1e6; % commutativity
rho = 1; % penalty for P
max_iters = 10;
links_type = 'min';
verbose = true;
delta = 1e-1;
forget = false;

%load('data.mat','A','Y2');
load('financial/m_X_air_OIL_CRYP2.mat','m_X');
Y2 = m_X;
[N,M] = size(Y2);
for n = 1:N
    y = Y2(n,:);
    Y2(n,:)=(Y2(n,:)-mean(y))/std(y);
end

O = N-2;
T = M-1;

X = Y2;
C = X*X'/M;
%C = C/norm(C,'fro')*100;
A = mbinarize(C,2);
%A = A/max(max(A));
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
    'max_iters',max_iters,'t0',M-T,'delta',delta,'forget',forget);


%[S_hat,out] = GSR_H(Co,Omega,reg,verbose);
%S_hat = S_hat/max(max(S_hat));
%err_GSRH = norm(S_hat-Ao,"fro")^2/nAo;
%fsc_GSRH = fscore(Ao,mbinarize(S_hat,2));

%GSR online
[AO_hat,outO] = GSR_O(Ctrain,X_test,Omega,reg,verbose);

%GSR H with online
[AOH_hat,outOH] = GSR_OH(Ctrain,X_test,Omega,reg,verbose);

%GSR H with offline
reg.max_iters = 100;
[AOH_off,outOH_off] = GSR_OH(Ctrain,X_test,Omega,reg,verbose);

%GSR offline
idx = find(A(:,1)==1); 
Omegaf = zeros(N);Omegaf(1,idx) = 1;Omegaf(idx,1) = 1;
Cf = X(:,1)*X(:,1)';
reg.max_iters = 100;
[AOf_hat,outOf] = GSR_O(Cf,X,Omegaf,reg,verbose);

%Compute the estimation error/fscore
allAO = outO.all_S;
allAOH = outOH.all_S;
allAOH_off = outOH_off.all_S;
allAOf = outOf.all_S;
%
fsc = zeros(T,2);
err = zeros(T,2);
for t = 1:T
   AhO = allAO(:,:,t);
   AhOH = allAOH(:,:,t);
   AhOH_off = allAOH_off(:,:,t);
   nAhOHoff = norm(AhOH_off,'fro');
   bin_AhOH = mbinarize(AhOH_off,2);
   AhOf = allAOf(s_n,s_n,t);
   nAhOf = norm(AhOf,'fro');
   bin_AhOf = mbinarize(AhOf,2);
   err(t,1) = (norm(AhO-AhOH_off,"fro")/nAhOHoff)^2;
   fsc(t,1) = fscore(bin_AhOH,mbinarize(AhO,2));
   err(t,2) = (norm(AhOH-AhOH_off,"fro")/nAhOHoff)^2;
   fsc(t,2) = fscore(bin_AhOH,mbinarize(AhOH,2));

   err(t,3) = (norm(AhO-AhOf,"fro")/nAhOf)^2;
   fsc(t,3) = fscore(bin_AhOf,mbinarize(AhO,2));
   err(t,4) = (norm(AhOH-AhOf,"fro")/nAhOf)^2;
   fsc(t,4) = fscore(bin_AhOf,mbinarize(AhOH,2));
   err(t,5) = (norm(AhOH_off-AhOf,"fro")/nAhOf)^2;
   fsc(t,5) = fscore(bin_AhOf,mbinarize(AhOH_off,2));

end

figure();
plot(err(:,3:4),LineWidth=2)
grid on
title('Error')
legend({'GSR-O f','GSR-OH f'})


%%
figure();
plot(err(20:end,3:4),LineWidth=3)
xlabel('Samples','FontSize',18,'FontWeight','bold','Interpreter','latex')
ylabel('Normalized error','FontSize',18,'FontWeight','bold','Interpreter','latex')
grid on
legend({'OnST','OnST-H'},'FontSize',18,'FontWeight','bold','Interpreter','latex')
ax = gca;
ax.FontSize = 18;

