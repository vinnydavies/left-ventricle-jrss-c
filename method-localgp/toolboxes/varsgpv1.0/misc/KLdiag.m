function out = KLdiag(mu0, Sigma0, mu1, Sigma1)
% KL divergecne between two Gaussians with diagonal covariance 
% matrices
%

N = size(Sigma0(:),1);

q = (mu1(:)-mu0(:))./Sigma1;
q = (mu1(:)-mu0(:))'*q;

out = 0.5*( sum(log(Sigma1)) - sum(log(Sigma0)) + ...
            sum(Sigma0./Sigma1) + q - N);