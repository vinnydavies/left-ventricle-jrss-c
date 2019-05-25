function [mustar, varstar, out3] = varsgpPredict(model, Xtest);
%
%
sigma2 = exp(2*model.Likelihood.logtheta);
if strcmp(model.indType, 'pseudoIns')
      Kmm = kernel(model.GP, model.Xu); 
      Knm = kernel(model.GP, model.X, model.Xu); 
      Kmstar = kernel(model.GP, model.Xu, Xtest);
elseif strcmp(model.indType, 'weights')
      Kmm = kernelWeights(model);
      Knm = kernelWeights(model, model.X);
      Kmstar = kernelWeights(model, Xtest)';
end

% add jitter to Kmm
Lm = chol(Kmm + model.jitter*eye(model.m));  % m x m: L_m^T where L_m is lower triangular   
invLm = Lm\eye(model.m);                           % m x m: L_m^{-T}                             
KnmInvLm = Knm*invLm;                        % n x m: K_nm L_m^{-T}                       

C = KnmInvLm'*KnmInvLm;                      % m x m: L_m^{-1}*Kmn*Knm*L_m^{-T}             
A = sigma2*eye(model.m) + C;                 % m x m: A = sigma2*I + L_m^{-1}*K_mn*K_nm*L_m^{-T}

% upper triangular Cholesky decomposition 
La = chol(A);                                       % m x m: L_A^T                     
invLa =  La\eye(model.m);                    % m x m: L_A^{-T}            

KstarminvL = Kmstar'*invLm; 
KstarminvLinvLa = KstarminvL*invLa;

% only diagonal  
Kstar = kernel(model.GP, Xtest, [], 1);  
varstar = Kstar - sum(KstarminvL.*KstarminvL,2) +  sigma2*sum(KstarminvLinvLa.*KstarminvLinvLa,2); 
      
switch model.Likelihood.type 
    case 'Gaussian'
    %        
      yKnmInvLmInvLa = (model.y'*KnmInvLm)*invLa;  % 1 x m: y^T*Knm*L_m^{-T}*L_A^{-T}  ---- O(m^2)
      mustar = KstarminvLinvLa*yKnmInvLmInvLa'; 
     
      if nargout == 3 
      %  % only diagonal  
      %  Kstar = kernel(model.GP, Xtest, [], 1);  
      %  varstar = Kstar - sum(KstarminvL.*KstarminvL,2) +  sigma2*sum(KstarminvLinvLa.*KstarminvLinvLa,2); 
      %else
        % full matrix  
        Kstar = kernel(model.GP, Xtest);
        out3 = Kstar - KstarminvL*KstarminvL' +  sigma2*(KstarminvLinvLa*KstarminvLinvLa');
        varstar = diag(out3);
      end
    %  
    case 'Probit'
    %         
      % update the mean of the truncated Gaussian 
      barZ = truncNormalStats(model.vardist.mu);
      
      yKnmInvLmInvLa = ((model.y.*barZ)'*KnmInvLm)*invLa;  % 1 x m: y^T*Knm*L_m^{-T}*L_A^{-T}  ---- O(m^2)
      mustar = KstarminvLinvLa*yKnmInvLmInvLa'; 
      
      %mustar1 = Kmstar'*((Kmm + Knm'*Knm)\(Knm'*(model.y.*barZ)));
      %a = (Kmm + Knm'*Knm)\Kmstar;
      %varstar1 = diag(Kstar) - sum(Kmstar.*(Kmm\Kmstar),1)'+sum(Kmstar.*a,1)';
      
      % probit likelihood 
      % take the probit probabilities for being in the class 1
      mu = mustar./sqrt(1 + varstar);
      out3 = probit(mu);
    %
end


