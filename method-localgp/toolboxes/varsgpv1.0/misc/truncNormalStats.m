function [a, b, entr] = truncNormalMoments(mu)
% moments of truncated normal with unit-variance 

logPhimu = probit(mu);
logNmu = -0.5*log(2*pi) - 0.5*(mu.*mu);

% mean 
lambda = logNmu - logPhimu;
lambda = exp(lambda);
a = mu + lambda; 

% variances
b = 1 - lambda.*(lambda +  mu); 

% compute also the entropies
entr = 0.5*log(2*pi) + 0.5*(a.*a + b -2*(a.*mu) + mu.^2)...
       + logPhimu;
