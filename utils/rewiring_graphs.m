function rew_out = rewiring_graphs(Mr,R,O,g_type,sig_type,links_type,pert_links,prms)
    %R is the number of rewirings
    %Mr is the number of samples for each rewiring
    verbose = false;
    prms.M = Mr;
    %generate a graph and R rewirings
    [A0, L] = generate_graph(g_type,prms);
    %Select hidden nodes
    [n_o, n_h] = select_hidden_nodes(links_type, O, L, zeros(size(L)));
    Ar =  gen_similar_graphs_hid(A0,R,pert_links,n_o,n_h);
    Omegar = zeros(O,O,R);
    Aor = zeros(O,O,R);
    Xor = zeros(O,Mr*R);
    for r = 1:R
        A = Ar(:,:,r);
        Ao = A(n_o,n_o);
        Aor(:,:,r) = Ao;
        idx = find(Ao(:,1)==1); 
        Omega = zeros(O);Omega(1,idx) = 1;Omega(idx,1) = 1;
        Omegar(:,:,r) = Omega;
        L = diag(sum(A))-A;
        [~,X,~,~] = generate_graph_signals(sig_type, L, prms, verbose);
        idx1 = (r-1)*Mr+1;
        idx2 = r*Mr;
        Xor(:,idx1:idx2) = X(n_o,:);

    end
    all_Aor = zeros(O,O,Mr*R);
    all_Omegar = zeros(O,O,Mr*R);
    for k = 1:R
        idx1 = (k-1)*Mr+1;
        idx2 = k*Mr;
        all_Aor(:,:,idx1:idx2) = repmat(Aor(:,:,k),[1,1,Mr]);
        all_Omegar(:,:,idx1:idx2) = repmat(Omegar(:,:,k),[1,1,Mr]);
    end

    rew_out.Ar = Ar;
    rew_out.Aor = all_Aor;
    rew_out.Omegar = all_Omegar;
    rew_out.Xor = reshape(Xor,[O,Mr*R]);
end