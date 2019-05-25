function [F, DF] = varsgpLikelihood(W, model, flag)
%
%


% PRELIMINARIES
if nargin == 2 
    flag = 1; 
end

% place W to the model structure
model = returnOptimizedParams(model, W, flag);
sigma2 = model.sigma2;

% COVARIANCE FUNCTION QUANTITIES needed in the sparse method: K_mm, Knm, diag(Knn)  
%
% inducing variable parameters are pseudo-inputs 
if strcmp(model.indType, 'pseudoIns') 
    % DXXu is used only in the 'se' kernel, otherwise is empty
    [model.Kmm, model.dXuXu]  = kernel(model.GP, model.Xu);
    [model.Knm, model.dXXu] = kernel(model.GP, model.X, model.Xu);
% inducing variables are linear combinations of training latent variables 
elseif strcmp(model.indType, 'weights')
%    
    % only inducing variables are optimized
    % this allows kernel quantities to be precomputed once 
    if flag==2 & isfield(model, 'KmmSq')
       model.Kmm = zeros(model.m, model.m);
       model.Knm = zeros(model.n, model.m);
       cnt = 0;
       for j=1:model.nIndParams
          W = sparse(diag(model.Xu(:,j)));
          model.Kmm = model.Kmm + W*(model.KmmSq(:,:,j)*W); 
          model.Knm = model.Knm + model.KnmAll(:,:,j)*W;
          % cross-terms for K_mm
          for k=j+1:model.nIndParams   
             cnt = cnt + 1;
             G = sparse(diag( model.Xu(:,j) ))*( model.KmmCr(:,:,cnt)*sparse(diag(model.Xu(:,k))) );
             model.Kmm = model.Kmm + (G + G');
          end 
       end
    else     
       [model.Kmm model.KmmSq model.KmmCr] = kernelWeights(model);
       [model.Knm model.KnmAll] = kernelWeights(model, model.X); 
    end
%   
end
%
model.diagKnn = kernel(model.GP, model.X, [], 1);
if model.GP.constDiag == 1
   model.TrKnn = model.n*model.diagKnn(1);
else
   model.TrKnn = sum(model.diagKnn);
end


% upper triangular Cholesky decomposition 
% (we add jitter to Kmm which implies jitter inducing variables
% however the matrix that is stored is jitter free. 
% The jitter-free matrix is used to compute more precise derivatives; see
% documentation)
Lm = chol(model.Kmm + model.jitter*eye(model.m)); % m x m: L_m^T where L_m is lower triangular   ---- O(m^3) 
invLm = Lm\eye(model.m);                          % m x m: L_m^{-T}                              ---- O(m^3)
KnmInvLm = model.Knm*invLm;                       % n x m: K_nm L_m^{-T}                         ---- O(n m^2)  !!!expensive!!!  

C = KnmInvLm'*KnmInvLm;                     % m x m: L_m^{-1}*Kmn*Knm*L_m^{-T}             ---- O(n m^2)  !!!expensive!!! 
A = sigma2*eye(model.m) + C;                % m x m: A = sigma2*I + L_m^{-1}*K_mn*K_nm*L_m^{-T}

% upper triangular Cholesky decomposition 
La = chol(A);                              % m x m: L_A^T                      ---- O(m^3)
invLa =  La\eye(model.m);                  % m x m: L_A^{-T}                   ---- O(m^3) 

% useful precomputed quantities
yKnmInvLm = (model.y'*KnmInvLm);           % 1 x m: y^T*Knm*L_m^{-T}           ---- O(n m)
yKnmInvLmInvLa = yKnmInvLm*invLa;          % 1 x m: y^T*Knm*L_m^{-T}*L_A^{-T}  ---- O(m^2)

% COMPUTE NEGATIVE LOWER BOUND
% F_0 + F_1 + F_2 in the report
F012 = - (model.n-model.m)*model.Likelihood.logtheta - 0.5*model.n*log(2*pi) - (0.5/sigma2)*(model.yy) - sum(log(diag(La)));
% F_3 in the report
F3 = (0.5/sigma2)*(yKnmInvLmInvLa*yKnmInvLmInvLa');
% Trace term: F_4 + F_5 in the report
TrK = - (0.5/sigma2)*(model.TrKnn  - sum(diag(C)) );
F = F012 + F3 + TrK;
% negative bound
F = - F;

% precomputations for the derivatives
invKmm = invLm*invLm';                    % m x m: K_mm^{-1} = L_m^{-T}*L_m^{-1}   ---- O(m^3)
invLmInvLa = invLm*invLa;                 % m x m: L_m^{-T}*L_A^{-T}               ---- O(m^3)
invA = invLmInvLa*invLmInvLa';            % m x m: A^{-1} = L_m^{-T}*L_A^{-T}*L_A^{-1}*L_m^{-1}  ---- O(m^3)
yKnmInvA = yKnmInvLmInvLa*invLmInvLa';    % 1 x m: y^T*K_nm*A^{-1}                 ---- O(m^2)

Tmm = sigma2*invA + yKnmInvA'*yKnmInvA;   % m x m: sigma2*A^{-1} + y^T*K_nm*A^{-1}*A^{-1}*K_mn*y

% auxiliary variable that is useful 
%for the sigma2 derivative
if strcmp(model.Likelihood.type, 'Gaussian') == 1
   %sigma2aux = sum(sum(model.Kmm.*Tmm)) + model.jitter*sum(diag(Tmm));
   yKnmInvLmInvLainvLa = yKnmInvLmInvLa*invLa';      % 1 x m: y^T*Knm*L_m^{-T}*L_A^{-T}*L_A^{-1}   ---- O(m^2)
   sigma2aux = sigma2*sum(sum(invLa.*invLa))  + yKnmInvLmInvLainvLa*yKnmInvLmInvLainvLa';
end

Tmm = invKmm - Tmm;                       % m x m: K_mm^{-1} - sigma2*A^{-1} -  A^{-1}*Kmn*y*y^T*Knm*A^{-1}
Tnm = model.Knm*Tmm;                      % n x m: Knm*(K_mm^{-1} - sigma2*A^{-1} - A^{-1}*Kmn*y*y^T*Knm*A^{-1})   ---- O(n m^2) !!!expensive!!!
Tmm = Tmm - (invLm*(C*invLm'))/sigma2;    % m x m: K_mm^{-1} - sigma2*A^{-1} -  (A^{-1}*Kmn*y*y^T*Knm*A^{-1}) - K_mm^{-1}*K_mn*K_nm*K_mm^{-1}/sigma2
Tnm = Tnm + (model.y*yKnmInvA);           % n x m: Knm*(K_mm^{-1} - sigma2*A^{-1} - A^{-1}*Kmn*y*y^T*Knm*A^{-1}) + y*y^T*Knm*A^{-1} 

% optimize over both inducing variable parameters and model 
% (kernel and likelihood) hyperparameters
if flag == 1
%
    % COMPUTE DERIVATIVES OF KERNEL HYPERPARAMETERS 
    [Dkern Dkernnm DTrKnn] = kernelSparseGradHyp(model, Tmm, Tnm); 
    Dkern = 0.5*Dkern + Dkernnm/model.sigma2 - (0.5/model.sigma2)*DTrKnn;
    
    % DERIVATIVES OF INDUCING VARIABLE PARAMETERS
    [DXu DXunm] = kernelSparseGradInd(model, Tmm, Tnm);
    DXu = DXu + DXunm/model.sigma2;
    
    % DERIVATIVES OF LIKELIHOOD HYPERPARAMETERS
    Dlik = [];
    if strcmp(model.Likelihood.type, 'Gaussian') == 1
       Dlik = - (model.n-model.m) + model.yy/sigma2 - 2*F3 - sigma2aux - 2*TrK;
    end

    % PUT EVERYTHING TOGETHER AND NEGATE
    DF = -[reshape(DXu', model.m*model.nIndParams, 1); Dkern; Dlik];
    %
% optimize over only inducing variable parameters   
elseif flag == 2
    % DERIVATIVES OF INDUCING VARIABLE PARAMETERS
    [DXu DXunm] = kernelSparseGradInd(model, Tmm, Tnm);
    DXu = DXu + DXunm/model.sigma2;
    
    % PUT EVERYTHING TOGETHER AND NEGATE
    DF = -[reshape(DXu', model.m*model.nIndParams, 1)]; 
    %
% optimize over only model (kernel and likelihood) hyperparameters   
elseif flag == 3 
    %
    % COMPUTE DERIVATIVES OF KERNEL HYPERPARAMETERS
    [Dkern Dkernnm DTrKnn] = kernelSparseGradHyp(model, Tmm, Tnm); 
    Dkern = 0.5*Dkern + Dkernnm/model.sigma2 - (0.5/model.sigma2)*DTrKnn;

    % DERIVATIVES OF LIKELIHOOD HYPERPARAMETERS
    Dlik = [];
    if strcmp(model.Likelihood.type, 'Gaussian') == 1
       Dlik = - (model.n-model.m) + model.yy/sigma2 - 2*F3 - sigma2aux - 2*TrK;
    end

    % PUT EVERYTHING TOGETHER AND NEGATE 
    DF = -[Dkern; Dlik];
end


