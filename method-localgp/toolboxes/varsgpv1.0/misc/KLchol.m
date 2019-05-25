function out = KLchol(mu0, L0, mu1, L1)
% KL divergecne between two Gaussians 
%

N = size(L0,1);

invL1 = L1\eye(N);
invL1L0 = invL1*L0; 
q = invL1*(mu1(:)-mu0(:));
out = sum(log(diag(L1))) - sum(log(diag(L0)))...
      + 0.5*( sum(sum(invL1L0.*invL1L0)) + q'*q - N);
  