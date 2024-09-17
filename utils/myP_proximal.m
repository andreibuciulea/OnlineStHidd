function Pout = myP_proximal(Pin,beta)
    O = size(Pin,1);
    %Compute the proximal for each column of P
    Pout = zeros(O);
    for n = 1:O
        p = Pin(:,n);
        Pout(:,n) = max(0, 1 - beta/norm(p))*p;
    end
end