function Dhyp = kernSparseGradHyp(model, Tmm, Tnm)
%
%

Dhyp = zeros(model.GP.nParams, 1);
%
switch model.GP.type 
    case 'se' 
       for d=1:model.GP.nParams   
          if d == 1 
             % derivatives over the kernel quantities K_mm, K_nm and
             % tr(K_nn) wrt the shared lengthscale
             % ell^2 = exp(2*log(ell)) and the deriv. is wrt log(ell)
             
             DKmm = model.Xu*model.Xu';
             dgXmm = diag(DKmm);
             DKmm = dgXmm*ones(1, model.m) + ones(model.m,1)*(dgXmm') - 2*DKmm;
             DKmm= (1/exp(2*model.GP.logtheta(1)))*DKmm;
             % cross-covariance matrix between training inputs and inducing variables
             DKnm = -2*model.X*model.Xu' + (sum(model.X.*model.X, 2))*ones(1, model.m)  + ones(model.n,1)*(sum(model.Xu.*model.Xu,2)');
             DKnm = (1/exp(2*model.GP.logtheta(1)))*DKnm;
             
             DKmm = DKmm.*model.Kmm;
             DKnm = DKnm.*model.Knm;
             DTrKnn = 0;
             
          else
             % derivatives over the kernel quantities K_mm, K_nm and
             % tr(K_nn) wrt kernel variance  sigmaf^2 = exp(2*log(sigmaf)) and the deriv. is wrt log(ell)
             DKmm = 2*model.Kmm;
             DKnm = 2*model.Knm;
             DTrKnn = 0;
             if strcmp(model.objectFunc, 'var') 
                DTrKnn = 2*model.TrKnn; 
             end
          end
          % derivative of the whole lower bound wrt. a kernel
          % hyperparameter
          Dhyp(d) = 0.5*sum(sum( DKmm.*Tmm )) + sum(sum( DKnm.*Tnm ))/model.sigma2 - (0.5/model.sigma2)*DTrKnn;
       end
    
    case 'seard' 
       for d=1:model.GP.nParams   
          if d <= model.D 
             model.Xu(:,d) = model.Xu(:,d)/exp(model.GP.logtheta(d));
             model.X(:,d) = model.X(:,d)/exp(model.GP.logtheta(d));
             % derivatives over the kernel quantities K_mm, K_nm and
             % tr(K_nn) wrt lengthscale
             % ell_d^2 = exp(2*log(ell_d)) and the deriv. is wrt log(ell_d)
             DKmm = ( (  model.Xu(:,d)*ones(1, model.m) - ones(model.m,1)*(model.Xu(:,d)') ).^2 ).*model.Kmm;
             DKnm = ( (  model.X(:,d)*ones(1, model.m) - ones(model.n,1)*(model.Xu(:,d)') ).^2  ).*model.Knm;
             DTrKnn = 0;
          else
             % derivatives over the kernel quantities K_mm, K_nm and
             % tr(K_nn) wrt kernel variance  sigmaf^2 = exp(2*log(sigmaf)) and the deriv. is wrt log(ell)
             DKmm = 2*model.Kmm;
             DKnm = 2*model.Knm;
             DTrKnn = 0;
             if strcmp(model.objectFunc, 'var') 
                DTrKnn = 2*model.TrKnn; 
             end
          end
          % derivative of the whole lower bound wrt. a kernel
          % hyperparameter
          Dhyp(d) = 0.5*sum(sum( DKmm.*Tmm )) + sum(sum( DKnm.*Tnm ))/model.sigma2 - (0.5/model.sigma2)*DTrKnn;
       end
end