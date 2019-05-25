function [F, DF] = varsgpLikelihood(W, model)
%
%


% extract inducing variables parameters
en = model.m*model.nIndParams;
model.Xu = reshape(W(1:en), model.m,  model.nIndParams);

% extra kernel hyperparameters
st = en + 1;
en = en + model.GP.nParams;
model.GP.logtheta = W(st:en); 

% extra likelihood hyperparameters
st = en + 1; 
en = en + model.Likelihood.nParams;
model.Likelihood.logtheta = W(st:en); 


%Xu = reshape(W(1:end-D-2),m,D);
%logtheta = W(end-D-1:end);
sigma2 = exp(2*model.Likelihood.logtheta);

% COVARIANCE MATRICES

%sigma2f = exp(2*logtheta(D+1));
%X = X ./( ones(n,1)*(exp(logtheta(1:D))') );
%Xu = Xu ./( ones(m,1)*(exp(logtheta(1:D))') );
%Kmm = Xu*Xu';
%dgKmm = diag(Kmm);
%Kmm = dgKmm*ones(1, m) + ones(m,1)*(dgKmm') - 2*Kmm;
%Kmm = sigma2f*exp(-0.5*Kmm);
%% add jitter to the inducing variables kernel matrix (this implies 'jitter' inducing variables)
%Kmm = Kmm + (jitter*sigma2f)*eye(m);
%Knm = -2*X*Xu' + (sum(X.*X,2))*ones(1,m)  + ones(n,1)*(sum(Xu.*Xu,2)');
%Knm = sigma2f*exp(-0.5*Knm);
%TrKnn = n*sigma2f;
 
[Kmm, Knm, TrKnn] = kernel(model.GP, model.X, model.Xu);

% upper triangular Cholesky decomposition 
Lm = chol(Kmm);  % L_m^T  
invLm = Lm\eye(model.m); % L_m^{-T}
KnmInvLm = Knm*invLm; % K_nm L_m^{-T}

C = KnmInvLm'*KnmInvLm; 
A = eye(model.m) + C/sigma2;  % A = I + L_m^{-1} * K_mn * K_nm * L_m^{-T} 

% upper triangular Cholesky decomposition 
La = chol(A);   % L_A^T
invLa =  La\eye(model.m); % L_A^{-T} 

% useful precomputed quantities
yKnmInvLm = (model.y'*KnmInvLm)/sigma2;  % 1 x m vetor : (y^T * Knm * L_m^{-T} )/sigma2
yKnmInvLmInvLa = yKnmInvLm*invLa;  % 1 x m vector (y^T * Knm * L_m^{-T} * L_A^{-T})/sigma2


% COMPUTE NEGATIVE LOWER BOUND
%
% F_0 + F_1 + F_2 in the report
F012 = - model.n*model.Likelihood.logtheta - 0.5*model.n*log(2*pi) - (0.5/sigma2)*(model.yy) - sum(log(diag(La)));
% F_3 in the report

F3 = 0.5*(yKnmInvLmInvLa*yKnmInvLmInvLa');

% Trace term: F_4 + F_5 in the report
TrK = - (0.5/sigma2)*(TrKnn  - sum(diag(C)) );

F = F012 + F3 + TrK;

% precomputations for the derivatives
invKmm = invLm*invLm';
invLmInvLa = invLm*invLa;
invA = invLmInvLa*invLmInvLa';
yKnmInvA = yKnmInvLmInvLa*invLmInvLa';

Tmm = invA + yKnmInvA'*yKnmInvA;
% compute some auxiliary variable that is useful for the sigma2 derivative 
sigma2aux = sum(sum(Kmm.*Tmm));
Tmm = invKmm - Tmm;
Tnm = Knm*Tmm;
Tmm = Tmm - (invLm*(C*invLm'))/sigma2;
Tnm = Tnm + (model.y*yKnmInvA);

% COMPUTE DERIVATIVES 
%
Dhyp = zeros(model.GP.nParams, model.Likelihood.nParams, 1);
for d=1:model.GP.nParams
%
    % KERNEL lengthscales
    %DKmm = ( (  Xu(:,d)*ones(1,m) - ones(m,1)*(Xu(:,d)') ).^2  ).*Kmm;
    %DKnm = ( (  X(:,d)*ones(1,m) - ones(n,1)*(Xu(:,d)') ).^2  ).*Knm; 
   
    [DKmm DKnm DTrKnn] = kernSparseGrad(model, d, Kmm, Knm, TrKnn);
   
    Dhyp(d) = 0.5*sum(sum( DKmm.*Tmm )) + sum(sum( DKnm.*Tnm ))/sigma2 - (0.5/sigma2)*DTrKnn; 
end

logtheta = model.GP.logtheta;

for d = model.nIndParams
    % INDUCING variables parameters
    DKnm = (( model.X(:,d)*ones(1,model.m) - ones(model.n,1)*(model.Xu(:,d)') )/exp(logtheta(d))).*Knm;
    DKmm = -(( ones(model.m,1)*(model.Xu(:,d)') - model.Xu(:,d)*ones(1,model.m) )/exp(logtheta(d))).*Kmm;       
    
    % all F terms together
    DXu(d,:) = sum( DKmm.*Tmm , 1 ) + sum( DKnm.*Tnm, 1)/sigma2;
%
end
% sigma2f = exp(2*log(sigmaf)) : derivatives wrt to log(sigmaf) 
%KnmInvLmInvLa = KnmInvLm*invLa;
%
%ok = KnmInvLmInvLa*yKnmInvLmInvLa';
%Dhyp(D+1) = - sum(sum( KnmInvLmInvLa.*KnmInvLmInvLa ))/sigma2  + 2*F3 - (ok'*ok)/sigma2 + 2*TrK;

%Dhyp(D+1) = sum(sum( Kmm.*Tmm )) + 2*sum(sum( Knm.*Tnm ))/sigma2  - (0.5/sigma2)*TrKnn;

% sigma2n = exp(2*log(sigman)) : derivatives wrt to log(sigman) 
Dhyp(end) = - (model.n-model.m) + model.yy/sigma2 - 2*F3 - sigma2aux - 2*TrK;

DF = -[reshape(DXu', model.m*model.nIndParams, 1); Dhyp];
F = - F;