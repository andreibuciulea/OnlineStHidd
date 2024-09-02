function res = compute_performance(in,T,Ao)

    allA = in.all_S;
    res = zeros(T,2);
    nAo = norm(Ao,'fro')^2;
    for t = 1:T
       A = allA(:,:,t);
       res(t,1) = norm(A-Ao,"fro")^2/nAo;
       res(t,2) = fscore(Ao,mbinarize(A,2));
    end
end