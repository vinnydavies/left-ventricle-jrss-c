function [model margLogL] = varsgpTrain(model, options)
%function [model margLogL] = varsgpTrain(model, options)
% 
% Descr : Optimizes the "variational lower bound" of the GP model over 
%         inducing variables parameters (either inducing inputs or linear 
%         weights; see documentation in pdf) and kernel hyperparameters. 
%         The routine works for standard GP regression and binary probit GP classification. 
%         It can also be called to optimize the DTC marginal likelihood instead of the 
%         variational lower bound.               
% Inputs:
%         * model: Structure containing the GP model; see varsgpCreate.
%         * options: Several options used in the optimization. options(1)
%           is the maximum number of the objective function evaluations. 
%           options(2) is the type of optimization: at the moment only scaled 
%           conjugate gradients is used and particularly the minimize.m function of
%           Carl Rasmussen. options(3) = 1,2,or 3; when 1 we optimize over both 
%           model (kernel and likelihood) hyperoarameters and inducing variables parameters, 
%           when 2 we optimize only over inducing variables parameters, when 3 we optimize 
%           over only model hyperparameters.   
% Outputs:  
%         * model: the model structure with the optimized parameters 
%         * margLogL:  the final value of the variational lower bound on the 
%           log marginal likelihood. 
%
% See also varsgpCreate, varsgpPredict
%
% Michalis  Titsias, 2010

FuncEval = -options(1);

% extract the vector of optimized parameters from the model structure
% and optimize the model 
switch model.Likelihood.type 
    case 'Gaussian'
    
      % extract the parameters to be optimized  
      W = extractOptimizedParams(model, options(3));
      
      % if you optimize only inducing variable parameters (not model 
      % hyperparameters), then precompute the krenel matrices 
      % sicne they do not contain adapted parameters 
      if (options(3) == 2 & strcmp(model.indType, 'weights')) 
          model.KmmSq = zeros(model.m, model.m, model.nIndParams);
          Tcross = factorial(model.nIndParams)/(factorial(2)*factorial(model.nIndParams-2)); 
          model.KmmCr = zeros(model.m, model.m, Tcross);
          model.KnmAll = zeros(model.n, model.m, model.nIndParams);          
          cnt = 0;       
          for j=1:model.nIndParams
             % square terms for K_mm 
             model.KmmSq(:,:,j) = kernel( model.GP, model.X(model.IndSubset(:,j),:) );
             % cross-covariance matrix between training inputs and inducing variables
             model.KnmAll(:,:,j) = kernel( model.GP, model.X,  model.X(model.IndSubset(:,j),:) );    
             for k=j+1:model.nIndParams   
             %   
               cnt = cnt + 1;
               model.KmmCr(:,:,cnt) = kernel( model.GP, model.X(model.IndSubset(:,j),:),  model.X(model.IndSubset(:,k),:) ); 
             %  
             end
          end
      end         
      
      % run the minimizer
      if strcmp(model.objectFunc, 'var') 
        [W, fX] = minimize(W, 'varsgpLikelihood', FuncEval, model, options(3));
      elseif strcmp(model.objectFunc, 'dtc')
        [W, fX] = minimize(W, 'dtcsgpLikelihood', FuncEval, model, options(3));
      end
      
      margLogL = -fX(end);
      
      % place back the optimized parameters in the model structure 
      model = returnOptimizedParams(model, W, options(3));
      
    case 'Probit'
     
      % place parameters in single vector  
      W = extractOptimizedParams(model, options(3));
      
      % initialize the exptected values under the variational  truncated Gaussians
      barZ = ones(model.n,1) + 0.01*randn(model.n,1);
      
      model = returnOptimizedParams(model, W, options(3));
      % intiliaze the mu parameters in the variational  truncated Gaussians
      model.vardist.mu = probitMu(model, barZ, options(3));
      
      % find the mean, variances, and entropy of the truncated Gaussian  
      [barZ, S, entr] = truncNormalStats(model.vardist.mu);
      
      
      % optimize parameters by running variational EM with two steps (see
      % papers and documentation)
      for it=1:options(1) 
      %
         % 1-STEP (keep fixed variational distribution psi and maximize
         % everything else)
         model1 = model;
         model1.y = model.y.*barZ; 
         
         model1.yy = model1.y'*model1.y;
         model1.vary = var(model1.y);
         
         % do few optimization steps
         if strcmp(model1.objectFunc, 'var') 
           [W, fX] = minimize(W, 'varsgpLikelihood', -6, model1, options(3));
         elseif strcmp(model.objectFunc, 'dtc')
           [W, fX] = minimize(W, 'dtcsgpLikelihood', -6, model1, options(3));
         end
         
         LogLBefUpPsiZ = -fX(end) - 0.5*sum(S(:)) + sum(entr(:));
   
         % 2-STEP : update the variational truncated Gaussian 
         %
         % update truncated Gaussians mean parameter
         model = returnOptimizedParams(model, W, options(3));
         model.vardist.mu = probitMu(model, barZ, options(3));
         
         % update the mean of the truncated Gaussian 
         [barZ, S, entr] = truncNormalStats(model.vardist.mu);
   
         % compute the lower bound
         model1 = model;
         model1.y = model.y.*barZ; 
         model1.yy = model1.y'*model1.y;
         model1.vary = var(model1.y);
         if strcmp(model1.objectFunc, 'var') 
           fX = varsgpLikelihood(W, model1, options(3));
         elseif strcmp(model.objectFunc, 'dtc')
           fX = dtcsgpLikelihood(W, model1, options(3));
         end
         
         margLogL = -fX - 0.5*sum(S(:)) + sum(entr(:));

         % print the lower bound of log marginal likelihood
         if (it>1)
            fprintf(1,'Iteration%4d  F %11.6f  Diffs %20.12f\n',it,margLogL,margLogL-oldF);
            %fprintf(1,'Iteration%4d  F %11.6f  Diffs %20.12f\n',it,margLogL(it),margLogL(it)-LogLBefUpPsiZ);
         else
            fprintf(1,'Iteration%4d  F %11.6f\n',it, margLogL);
         end
         oldF = margLogL;
      
      %
      end      
      % place back the optimized parameters in the model structure
      model = returnOptimizedParams(model, W, options(3));
end
  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% auxiliary function used in probit classification 
function mu = probitMu(model, barZ, flag)
%
%
% inducing variable parameters are pseudo-inputs 
if strcmp(model.indType, 'pseudoIns')
   model.Kmm = kernel(model.GP, model.Xu);
   model.Knm = kernel(model.GP, model.X, model.Xu);
% inducing variable parameters are linear weights between training latent variables    
elseif strcmp(model.indType, 'weights')
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
end 
   
% add jitter to Kmm
Lm = chol(model.Kmm + model.jitter*eye(model.m));  % m x m: L_m^T where L_m is lower triangular   
invLm = Lm\eye(model.m);                           % m x m: L_m^{-T}                             
KnmInv = model.Knm*invLm;                   % n x m: K_nm L_m^{-T}                       
A = KnmInv'*KnmInv;                          % m x m: L_m^{-1}*Kmn*Knm*L_m^{-T}             
A = eye(model.m) + A;                                 % m x m: A = I + L_m^{-1}*K_mn*K_nm*L_m^{-T}
% upper triangular Cholesky decomposition 
La = chol(A);                                                % m x m: L_A^T                     
invLa =  La\eye(model.m);                            % m x m: L_A^{-T}            

KnmInv = KnmInv*invLa;                             
mu = (KnmInv*(KnmInv'*(model.y.*barZ)));            
mu = model.y.*mu;                                       % Y*Knm*inv(Kmm + Kmn*Knm)*Kmn*Y*barZ 

% old  code
% add jitter to Kmm
%model.Kmm = model.Kmm + model.jitter*eye(model.m);
%YKnm = (model.y*ones(1, model.m)).*model.Knm;
%A = model.Kmm + (model.Knm'*model.Knm);        % A = K_mm + Kmn*Knm 
%La = chol(A);  
%invLa = La\eye(model.m);                       % L_A^{-T} 
%YKnmInvLa = YKnm*invLa;                        % Y*Knm*L_A^{-T}  
%mu = (YKnmInvLa*(YKnmInvLa'*barZ));            % Y*Knm*inv(A)*Kmn*Y*barZ 

