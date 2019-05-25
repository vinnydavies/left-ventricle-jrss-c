function [F, DF] = vargpLikelihood(W, X, y, m)
%
%

% number of examples and dimension of input space
[n, D] = size(X);
jitter = 1e-7;

% extract pseudo inputs and hyperparameters
Xu = reshape(W(1:end-D-2),m,D);
logtheta = W(end-D-1:end);
sigma2 = exp(2*logtheta(D+2));
sigma = exp(logtheta(D+2));
sigma2f = exp(2*logtheta(D+1));

X = X ./( ones(n,1)*(exp(logtheta(1:D))') );
Xu = Xu ./( ones(m,1)*(exp(logtheta(1:D))') );

% COVARIANCES MATRICES
% Kmm = K(Xu,Xu)
Kmm = Xu*Xu';
dgKmm = diag(Kmm);
Kmm = dgKmm*ones(1, m) + ones(m,1)*(dgKmm') - 2*Kmm;
Kmm = sigma2f*exp(-0.5*Kmm);
% add jitter to the inducing variables kernel matrix (this implies 'jitter' inducing variables)
Kmm = Kmm + (jitter*sigma2f)*eye(m);
%Knm = K(X,Xu)
Knm = -2*X*Xu' + (sum(X.*X,2))*ones(1,m)  + ones(n,1)*(sum(Xu.*Xu,2)');
Knm = sigma2f*exp(-0.5*Knm);
% TrKnn = tr(Knn)
TrKnn = n*sigma2f;

C = (Knm'*Knm);
A = sigma2*Kmm + C; 

Lm = chol(Kmm)';
%invLm = Lm\eye(m);
%invKmm = invLm'*invLm;
invSigmaLm = (Lm\eye(m))/sigma;
invSigma2Kmm = invSigmaLm'*invSigmaLm;

C1 = Knm*(invSigmaLm');

La = chol(A)';
invLa =  La\eye(m);
invA = invLa'*invLa;

% useful precomputed quantities
KmnY = Knm'*y;  % m x 1 vetor : Kmn y
invLaY = invLa*KmnY; 

%KnmInvA = Knm*invA; % n x m matrix: Knm * inv(A)
%KnmInvKmm = Knm*invKmm;  % n x m matrix:  Knm * inv(Kmm)
%KnmInvKmm2 = KnmInvKmm'*KnmInvKmm; % m x m matrix: inv(Kmm) * Kmn * Knm * inv(Kmm)
yKnmInvA = KmnY'*invA; % 1 x m vector: y^T * Knm inv(A) 
%yKnmInvAKmn = yKnmInvA*Knm';
%

yy = y'*y; 

% COMPUTE NEGATIVE LOWER BOUND
% F_0 + F_1 + F_2 in the report
F012 = - (n-m)*logtheta(D+2) - 0.5*n*log(2*pi) - (0.5/sigma2)*(yy) + sum(log(diag(Lm))) - sum(log(diag(La)));
% F_3 in the report

F3 = (0.5/sigma2)*(invLaY'*invLaY);
%F3 = (0.5/sigma2)*(yKnmInvA*KmnY);

% F_4 + F_5 in the report
%TrK = - (0.5/sigma2)*TrKnn  + 0.5*sum(sum(C.*invSigma2Kmm));
TrK = - (0.5/sigma2)*TrKnn  + 0.5*sum(sum(C1.*C1));

F = F012 + F3 + TrK;

% precomputations for the derivatives
yKnmInvAsigma = yKnmInvA/sigma;
Tmm = invSigma2Kmm - invA - (yKnmInvAsigma'*yKnmInvAsigma);
Tnm = Knm*Tmm;

%Tmm = Tmm - invSigma2Kmm*(C*invSigma2Kmm);
C1 = C1*invSigmaLm; 
Tmm = Tmm - (C1'*C1);

%Tmm = Tmm - 0.5*( invSigma2Kmm*(C*invSigma2Kmm) + (invSigma2Kmm*C)*invSigma2Kmm  ) ;
Tnm = Tnm + (y*(yKnmInvA/sigma2));

% COMPUTE DERIVATIVES  
Dhyp = zeros(D+2,1);
for d=1:D
%
    % KERNEL lengthscales
    DKmm = ( (  Xu(:,d)*ones(1,m) - ones(m,1)*(Xu(:,d)') ).^2  ).*Kmm;
    DKnm = ( (  X(:,d)*ones(1,m) - ones(n,1)*(Xu(:,d)') ).^2  ).*Knm; 
    
    Dhyp(d) = (0.5*sigma2)*sum(sum( DKmm.*Tmm )) + sum(sum( DKnm.*Tnm )); 
    
    %% F_1 + F_2 + F_3
    %t1 = 0.5*sum(sum( invKmm.*DKmm)) - (0.5*sigma2)*sum(sum(invA.*DKmm))...
    %     - sum(sum( KnmInvA.*DKnm )) ...
    %     + (1/sigma2)*((y'*DKnm)*yKnmInvA') ...
    %     - 0.5*(yKnmInvA*(DKmm*yKnmInvA')) - (1/sigma2)*(yKnmInvAKmn*(DKnm*yKnmInvA'));
    %
    %% F_5 term     
    %t2 = (0.5/sigma2)*sum(sum( - DKmm.*KnmInvKmm2)) + (1/sigma2)*sum(sum(DKnm.*KnmInvKmm)); 
    %
    %Dhyp(d) = t1 + t2;
    
    % INDUCING variables parameters
    DKnm = (( X(:,d)*ones(1,m) - ones(n,1)*(Xu(:,d)') )/exp(logtheta(d))).*Knm;
    DKmm = -(( ones(m,1)*(Xu(:,d)') - Xu(:,d)*ones(1,m) )/exp(logtheta(d))).*Kmm;       
    
    % all F terms together
    DXu(d,:) = sigma2*sum( DKmm.*Tmm , 1 ) + sum( DKnm.*Tnm, 1);
    
    %DXu(d,:) = sum( invKmm.*DKmm , 1 )  - sigma2*sum(invA.*DKmm,1) -  sum(KnmInvA.*DKnm,1) ...
    %           + (1/sigma2)*((  (y'*DKnm) - sigma2*(yKnmInvA*DKmm) - yKnmInvAKmn*DKnm  ).*yKnmInvA ...
    %           +  sum( - DKmm.*KnmInvKmm2, 1) + sum(DKnm.*KnmInvKmm, 1) );     
%
end

% sigma2f = exp(2*log(sigmaf)) : derivatives wrt to log(sigmaf) 
Dhyp(D+1) = - sum(sum(invA.*C))  + 2*F3 - (yKnmInvA*(C*yKnmInvA'))/sigma2 + 2*TrK;

% sigma2n = exp(2*log(sigman)) : derivatives wrt to log(sigman) 
Dhyp(D+2) = - (n-m) + yy/sigma2 - 2*F3 - ...
            (yKnmInvA*(Kmm*yKnmInvA')) - sigma2*sum(sum(Kmm.*invA)) - 2*TrK;

DF = -[reshape(DXu', m*D, 1); Dhyp];
F = - F;