function [K dXXu] = kernel(GP, X, Xu, flag)
%function K = kernel(GP, X, Xu, flag)
%
% Descr : Compute a kernel matrix for specific inputs and covariance 
%         function. Three modes are possible. Calling with 2 arguments, 
%         ie. kernel(GP, X), the covariance matrix K(X,X) is computed. Calling with
%         3 arguments, ie. kernel(GP, X, Xu), then K(X,Xu) is computed. Calling with 
%         4 arguments, ie. kernel(GP, X, Xu, flag), then only the diagonal
%         of K(X,X) is computed and when this diagonal is constant, only single scalar 
%         with the common diagonal element of K(X,X) is returned. 
% Inputs:
%         * GP: Structure containing the GP prior. This includes the type
%           of the covariance function, the hyperpapaeameters etc.   
%         * X: n x D matrix with inputs (each per row)
%         * Xu: m x D matrix inputs 
%         * flag: a dummy variable, any value will have the same effect 
%           since the function detects that 4 input arguments are given  
% Outputs:  
%         * K: the computed kernel matrix  
%
% Supported covariance functions: squared exponential ("se"), ARD squared
%         exponential ("seard") 
%
% Notes: Sums of different covariance function is not currently supported 
%
% See also  kernelWeights
%
% Michalis  Titsias, 2010
  
dXXu = [];
switch GP.type
%    
    case 'se'
      [n, D] = size(X);
      sigma2f = exp(2*GP.logtheta(2));
      % square covariance matrix
      if nargin == 2
      %  
          K = X*X';
          dgK = diag(K);
          %K = dgK*ones(1, n) + ones(n, 1)*(dgK') - 2*K;
          K =  bsxfun(@plus, dgK, dgK') - 2*K;
          if nargout == 2
              dXXu = K;
          end
          K = sigma2f*exp(-(0.5/exp(2*GP.logtheta(1)))*K);
      % cross-covariance matrix between training inputs and inducing variables   
      elseif nargin == 3
      %    
          m = size(Xu, 1); 
          %K = -2*X*Xu' + (sum(X.*X,2))*ones(1, m)  + ones(n,1)*(sum(Xu.*Xu,2)');
          K = -2*X*Xu' + bsxfun(@plus, sum(X.*X,2), sum(Xu.*Xu,2)');
          if nargout == 2
              dXXu = K;
          end
          K = sigma2f*exp(-(0.5/exp(2*GP.logtheta(1)))*K);
      % compute only the diagonal of kernel matrix for X
      elseif nargin == 4
      %    
          % Since the diagonal is constant then you only have to 
          % store and compute one elements 
          K = sigma2f; 
      end
    %   
    case 'seard'      
    %    
      [n, D] = size(X);    
      sigma2f = exp(2*GP.logtheta(D+1));
      % square covariance matrix
      if nargin == 2
      % 
          X = X./( ones(n,1)*(exp(GP.logtheta(1:D))') );      
          % covariance matrix over the inducing variables
          K = X*X';
          dgK = diag(K);
          K = dgK*ones(1, n) + ones(n, 1)*(dgK') - 2*K;
          K = sigma2f*exp(-0.5*K);
      % cross-covariance matrix between training inputs and inducing variables   
      elseif nargin == 3
      %    
          m = size(Xu,1);
          X = X ./( ones(n,1)*(exp(GP.logtheta(1:D))') );
          Xu = Xu ./( ones(m,1)*(exp(GP.logtheta(1:D))') );
          
          % cross-covariance matrix between training inputs and inducing variables
          K = -2*X*Xu' + (sum(X.*X,2))*ones(1,m)  + ones(n,1)*(sum(Xu.*Xu,2)');
          K = sigma2f*exp(-0.5*K);  
      % comptue only the diagonal of kernel matrix for X
      elseif nargin == 4
      %    
          % Since the diagonal is constant then you only have to 
          % store and compute one element         
          K = sigma2f;
      end
%
end
