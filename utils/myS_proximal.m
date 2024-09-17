function S = myS_proximal(S,Omega,alpha)
    %Omega is a matrix of size OxO whose nonzero entries are the known
    %links
    % Ensure the matrix is symmetric by averaging with its transpose
    S = (S + S') / 2;
    
    % Apply soft thresholding
    S = max(0, S - alpha);
    
    % Set diagonal elements to zero
    S(logical(eye(size(S)))) = 0;
    
    % Use the known links
    S(Omega~=0) = Omega(Omega~=0); 
    
    S = S/max(max(S));

end