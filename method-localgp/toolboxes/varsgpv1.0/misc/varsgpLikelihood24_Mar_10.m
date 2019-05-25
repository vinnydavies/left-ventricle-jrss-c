function [F, DF] = varsgpLikelihood(W, model)
%
%


%%%%%% extract inducing variables parameters
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

sigma2 = exp(2*model.Likelihood.logtheta);
%%%%%%


% COVARIANCE FUNCTION QUANTITIES : K_mm, Knm, tr(Knn)  
[Kmm, Knm, TrKnn] = kernel(model.GP, model.X, model.Xu);

% upper triangular Cholesky decomposition 
Lm = chol(Kmm);                            % m x m: L_m^T where L_m is lower triangular    
invLm = Lm\eye(model.m);                   % m x m: L_m^{-T}
KnmInvLm = Knm*invLm;                      % n x m: K_nm L_m^{-T}

C = KnmInvLm'*KnmInvLm;                    % 
A = eye(model.m) + C/sigma2;               % m x m: A = I + L_m^{-1} * K_mn * K_nm * L_m^{-T} 

% upper triangular Cholesky decomposition 
La = chol(A);                              % m x m: L_A^T
invLa =  La\eye(model.m);                  % m x m: L_A^{-T} 

% useful precomputed quantities
yKnmInvLm = (model.y'*KnmInvLm)/sigma2;    % 1 x m: (y^T * Knm * L_m^{-T} )/sigma2
yKnmInvLmInvLa = yKnmInvLm*invLa;          % 1 x m: (y^T * Knm * L_m^{-T} * L_A^{-T})/sigma2


% COMPUTE NEGATIVE LOWER BOUND
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
% auxiliary variable that is useful for the sigma2 derivative 
sigma2aux = sum(sum(Kmm.*Tmm));
Tmm = invKmm - Tmm;
Tnm = Knm*Tmm;
Tmm = Tmm - (invLm*(C*invLm'))/sigma2;
Tnm = Tnm + (model.y*yKnmInvA);

% COMPUTE DERIVATIVES OF KERNEL HYPERPARAMETERS
Dhyp = zeros(model.GP.nParams + model.Likelihood.nParams, 1);
for d=1:model.GP.nParams
%   
    [DKmm DKnm DTrKnn] = kernSparseGrad(model, d, Kmm, Knm, TrKnn);
    Dhyp(d) = 0.5*sum(sum( DKmm.*Tmm )) + sum(sum( DKnm.*Tnm ))/sigma2 - (0.5/sigma2)*DTrKnn; 
%    
end

% DERIVATIVES OVER INDUCING VARIABLE PARAMETERS
logtheta = model.GP.logtheta;
for d = 1:model.nIndParams
%    
    DKnm = (( model.X(:,d)*ones(1,model.m) - ones(model.n,1)*(model.Xu(:,d)') )/exp(2*logtheta(d))).*Knm;
    DKmm = -(( ones(model.m,1)*(model.Xu(:,d)') - model.Xu(:,d)*ones(1,model.m) )/exp(2*logtheta(d))).*Kmm;       
    
    DXu(d,:) = sum( DKmm.*Tmm , 1 ) + sum( DKnm.*Tnm, 1)/sigma2;
%
end

% DERIVATIVES OF LIKELIHOOD HYPERPARAMETERS 
Dhyp(end) = - (model.n-model.m) + model.yy/sigma2 - 2*F3 - sigma2aux - 2*TrK;

% PUT EVERYTHING TOGETHER AND NEGATE 
DF = -[reshape(DXu', model.m*model.nIndParams, 1); Dhyp];
F = - F;