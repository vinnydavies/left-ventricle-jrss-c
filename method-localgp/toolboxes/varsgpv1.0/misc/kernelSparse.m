function [Kmm, Knm, diagKnn] = kernelSparse(GP, X, Xu)
% It wil compute all  kernel quntities needed in the variational 
% sparse approxiamtion 
%
%

[n, D] = size(X);
m = size(Xu,1);

if strcmp(GP.indType, 'pseudoIns')
%    
  switch GP.type
  %    
    case 'se'              
      sigma2f = exp(2*GP.logtheta(2));
      % covariance matrix over the inducing variables
      Kmm = Xu*Xu';
      dgKmm = diag(Kmm);
      Kmm = dgKmm*ones(1, m) + ones(m,1)*(dgKmm') - 2*Kmm;
      Kmm = sigma2f*exp(-(0.5/exp(2*GP.logtheta(1)))*Kmm);
      
      % cross-covariance matrix between training inputs and inducing variables
      Knm = -2*X*Xu' + (sum(X.*X,2))*ones(1,m)  + ones(n,1)*(sum(Xu.*Xu,2)');
      Knm = sigma2f*exp(-(0.5/exp(2*GP.logtheta(1)))*Knm);
      
      % Since the diagonal is constant then you only have to 
      % store and compute one elements 
      diagKnn(1) = sigma2f;
      
      %   
    case 'seard'      
      
      X = X ./( ones(n,1)*(exp(GP.logtheta(1:D))') );
      Xu = Xu ./( ones(m,1)*(exp(GP.logtheta(1:D))') );

      sigma2f = exp(2*GP.logtheta(D+1));
      
      % covariance matrix over the inducing variables
      Kmm = Xu*Xu';
      dgKmm = diag(Kmm);
      Kmm = dgKmm*ones(1, m) + ones(m,1)*(dgKmm') - 2*Kmm;
      Kmm = sigma2f*exp(-0.5*Kmm);
      
      % cross-covariance matrix between training inputs and inducing variables
      Knm = -2*X*Xu' + (sum(X.*X,2))*ones(1,m)  + ones(n,1)*(sum(Xu.*Xu,2)');
      Knm = sigma2f*exp(-0.5*Knm);
      
      % Since the diagonal is constant then you only have to 
      % store and compute one element 
      diagKnn(1) = sigma2f;
      %
  end
%    
elseif strcmp(model.indType, 'weights')
%
 switch GP.type
  %    
    case 'se'  
      sigma2f = exp(2*GP.logtheta(2));
      
      % covariance matrix over the inducing variables
      Kmm = zeros(m, m);     
      % square terms 
      for j=1:model.nIndParams
      %    
          F = X(IndSubset(:,j),:)*X(IndSubset(:,j),:)';
          dgF = diag(F);
          F = dgF*ones(1, m) + ones(m,1)*(dgF') - 2*F;
          F = sigma2f*exp(-(0.5/exp(2*GP.logtheta(1)))*F);
          W = sparse(diag(Xu(:,j)));
          Kmm = Kmm + W*(F*W);
      %    
      end    
      
      % cross terms
      for j=1:model.nIndParams
         for k=j+1:model.nIndParams   
         %    
            F = -2*X(IndSubset(:,j),:)*X(IndSubset(:,k),:)' + ...
                (sum(X(IndSubset(:,j),:).*X(IndSubset(:,j),:),2))*ones(1,m) + ...
                ones(m,1)*(sum(X(IndSubset(:,k),:).*X(IndSubset(:,k),:),2)');
             
            F = sigma2f*exp(-(0.5/exp(2*GP.logtheta(1)))*F);  
            F = sparse(diag(Xu(:,j)))*(F*sparse(diag(Xu(:,k))));
            Kmm = Kmm + (F + F');
         %   
         end
      end
          
      % cross-covariance matrix between training inputs and inducing variables
      Knm = zeros(n, m);
      for j=1:model.nIndParams
      %    
         F = -2*X*X(IndSubset(:,j),:)' + (sum(X.*X,2))*ones(1,m) + ones(n,1)*(sum(X(IndSubset(:,j),:).*X(IndSubset(:,j),:),2)');
         F = sigma2f*exp(-(0.5/exp(2*GP.logtheta(1)))*F);
         Knm = Knm + F*sparse(diag(Xu(:,j)));
      %   
      end
      
      % Since the diagonal is constant then you only have to 
      % store and compute one elements 
      diagKnn(1) = sigma2f;
      
    case 'seard'      
      
      X = X ./( ones(n,1)*(exp(GP.logtheta(1:D))') );
      Xu = Xu ./( ones(m,1)*(exp(GP.logtheta(1:D))') );

      sigma2f = exp(2*GP.logtheta(D+1));
      
      % covariance matrix over the inducing variables
      Kmm = Xu*Xu';
      dgKmm = diag(Kmm);
      Kmm = dgKmm*ones(1, m) + ones(m,1)*(dgKmm') - 2*Kmm;
      Kmm = sigma2f*exp(-0.5*Kmm);
      
      % cross-covariance matrix between training inputs and inducing variables
      Knm = -2*X*Xu' + (sum(X.*X,2))*ones(1,m)  + ones(n,1)*(sum(Xu.*Xu,2)');
      Knm = sigma2f*exp(-0.5*Knm);
      
      % Since the diagonal is constant then you only have to 
      % store and compute one element 
      diagKnn(1) = sigma2f;
      %
  end 
%
end