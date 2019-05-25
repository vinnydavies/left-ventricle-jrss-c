function [f, dfX] = Var_lik(W, X, y, m)
%
%

% number of examples and dimension of input space
[n, D] = size(X);
jitter = 1e-7;

% extract pseudo inptus and hyperparameters
Xu = reshape(W(1:end-D-2),m,D);
logtheta = W(end-D-1:end);
sigma2n = exp(2*logtheta(D+2));
sigmaf = exp(2*logtheta(D+1));

X = X ./ repmat(exp(logtheta(1:D))',n,1);
Xu = Xu ./ repmat(exp(logtheta(1:D))',m,1);

% write the noise-free covariance of size n x m
Kmm = Xu*Xu';
Kmm = repmat(diag(Kmm),1,m) + repmat(diag(Kmm)',m,1) - 2*Kmm;
Kmm = sigmaf*exp(-0.5*Kmm);
Kmm = Kmm + (jitter*1.8828)*eye(m);% + (abs(mean(diag(Kmm)))*jitter)*eye(m);
Knm = -2*Xu*X' + repmat(sum(X.*X,2)',m,1) + repmat(sum(Xu.*Xu,2),1,n);
Knm = sigmaf*exp(-0.5*Knm');

%Knm = sigmaf*exp(-0.5*sq_dist(X',Xu'));
%Kmm = sigmaf*exp(-0.5*sq_dist(Xu',Xu'));
%% add little jitter to Kmm part
%Kmm = Kmm1+1e-8*eye(m);

Lkmm = chol(Kmm);
%invLkmm = Lkmm\eye(m);
Cnm1 = Knm/Lkmm;
%Cnm1 = Knm*invLkmm;
Cmnmn = Cnm1'*Cnm1;
Lm = chol(sigma2n*eye(m) + Cmnmn);
invLm = Lm\eye(m);
Pnm1 = Cnm1/Lm;
%Pnm1 = Cnm1*invLm;

%Pmnmn = Cmnmn*invLm;
Pmnmn = (Lm'\Cmnmn)';
%sum(sum(Pmnmn1-Pmnmn))

bet = Pnm1'*y;

%invLkmmLm = invLkmm*invLm; 
%wm1 = invLkmmLm*bet;
LmLkmm = Lm*Lkmm;
wm1 = (LmLkmm)\bet;

%sum(abs(wm1-wm2))
%return
invQt1 = y - Pnm1*bet;

logdetQ = (n-m)*2*logtheta(D+2) + 2*sum(log(diag(Lm)));
f = 0.5*logdetQ +  (0.5/sigma2n)*(y'*y - bet'*bet)  + 0.5*n*log(2*pi);
% above: it should be removed for the SOReg approximation 
TrK = + (0.5/sigma2n)*(n*sigmaf - sum(diag(Cmnmn)));
f = f + TrK;
% below: it should be removed for the SOReg approximation  

% compute derivatives
df = zeros(D+2,1);

aux = sum(sum(invLm.*Pmnmn));
Pmnmn = (Lkmm\Cmnmn)';
BB1 = Lkmm\Pmnmn; 
BB1 = - BB1/sigma2n - wm1*wm1';
BB1 = BB1 + LmLkmm\(Lm'\Pmnmn);
BB1 = Kmm.*BB1;

Cnm1 = (Lkmm\Cnm1')';
Cnm1 = Cnm1 - sigma2n*(LmLkmm\Pnm1')';
Pnm1 = repmat(invQt1,1,m).*repmat(wm1',n,1);

Cnm1 = (Cnm1 + Pnm1).*Knm;
Pnm1 = Pnm1.*Knm;
%
for d=1:D
    %
    Knm = -sq_dist(X(:,d)',Xu(:,d)');
    Kmm = -sq_dist(Xu(:,d)',Xu(:,d)');
 
    df(d) = sum(sum(Knm.*Cnm1))/sigma2n + 0.5*sum(sum(Kmm.*BB1));
      
    % pseudo inputs derivatives
    Knm = -((repmat(Xu(:,d)',n,1)-repmat(X(:,d),1,m))/exp(logtheta(d)));
    Kmm = -((repmat(Xu(:,d)',m,1)-repmat(Xu(:,d),1,m))/exp(logtheta(d)));       
    
    dXu(:,d) = (sum(Knm.*Cnm1,1)/sigma2n + sum(Kmm.*BB1,1))'; 
    %
end
%
dXu = -dXu;
df(D+1) = -sum(sum(Pnm1))/sigma2n+aux;
df(D+2) = (n-aux) - (invQt1'*invQt1)/sigma2n;

% above: it should be removed for the SOReg approximation
df(D+1) = df(D+1) + 2*TrK;
df(D+2) = df(D+2) - 2*TrK; 
% below: it should be removed for the SOReg approximation

dfX = [reshape(dXu, m*D, 1); df];
