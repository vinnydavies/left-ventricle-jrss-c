function [logtheta xb mu s2 margLogL, elapsedTrain, elapsedPred] = runSnelsoncode(logtheta, x, y, xtest, xb_init, Iterations, hypflag)
%
%
%

[N,dim] = size(x);
M = size(xb_init,1);

%if hypflag == 1
%% initialize hyperparameters sensibly (see spgp_lik for how
%% the hyperparameters are encoded)
%hyp_init(1:dim,1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
%hyp_init(dim+1,1) = log(var(y,1)); % log size 
%hyp_init(dim+2,1) = log(var(y,1)/4); % log noise
%else
hyp_init(1:dim,1) = -2*logtheta(1:dim); % log 1/(lengthscales)^2
hyp_init(dim+1,1) = 2*logtheta(dim+1); % log size 
hyp_init(dim+2,1) = 2*logtheta(dim+2); % log noise   
%end  

w_init = [reshape(xb_init,M*dim,1);hyp_init];

% optimize hyperparameters and pseudo-inputs
tic; 
if hypflag == 1
  [w,f] = minimize(w_init,'spgp_lik',-Iterations,y,x,M);
elseif hypflag == 2
  [w,f] = minimize(w_init,'spgp_lik_nohyp',-Iterations,y,x,M);
elseif hypflag == 3
  [w,f] = minimize(w_init,'spgp_lik_noind',-Iterations,y,x,M);
else
  w = w_init;
end
elapsedTrain=toc;

% extract hyperparameters
xb = reshape(w(1:M*dim,1),M,dim);
hyp = w(M*dim+1:end,1);
  
margLogL = -spgp_lik(w,y,x,M);

% PREDICTION
tic; 
[mu,s2] = spgp_pred(y,x,xb,xtest,hyp);
% if you want predictive variances to include noise variance add noise:
s2 = s2 + exp(hyp(end));
elapsedPred=toc;

% transform back parameters
logtheta(1:dim,1) = -0.5*hyp(1:dim,1);
logtheta(dim+1) = 0.5*hyp(dim+1,1);
logtheta(dim+2) = 0.5*hyp(dim+2,1); 
