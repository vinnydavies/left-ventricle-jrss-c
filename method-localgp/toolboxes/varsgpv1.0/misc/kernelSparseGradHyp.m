function [Dhypmm Dhypnm DTrKnn] = kernSparseGradHyp(model, Tmm, Tnm)
%
%

Dhypmm = zeros(model.GP.nParams, 1);
Dhypnm = zeros(model.GP.nParams, 1);
DTrKnn = zeros(model.GP.nParams, 1);

%
if strcmp(model.indType, 'pseudoIns')
%
    switch model.GP.type 
    case 'se' 
       model.Knm = model.Knm.*Tnm;
       model.Kmm = model.Kmm.*Tmm;
       for d=1:model.GP.nParams   
          if d == 1 
             % derivatives over the kernel quantities K_mm, K_nm and
             % tr(K_nn) wrt the shared lengthscale
             % ell^2 = exp(2*log(ell)) and the deriv. is wrt log(ell)    
             % distance between  inducing inputs
             % (it should be have precomputed in the model.dXXu)   
             %DKmm = model.Xu*model.Xu';
             %dgXmm = diag(DKmm);
             %DKmm = dgXmm*ones(1, model.m) + ones(model.m,1)*(dgXmm') - 2*DKmm;
             
             % distance between training inputs and inducing inputs
             % (it should be have precomputed in the model.dXXu)
             %DKnm = -2*model.X*model.Xu' + model.XX*ones(1, model.m)  + ones(model.n,1)*(sum(model.Xu.*model.Xu,2)');
             
             % derivative of the hyperparameters
             Dhypmm(d) = sum(sum( model.dXuXu.*model.Kmm ))/exp(2*model.GP.logtheta(1));
             Dhypnm(d) = sum(sum( model.dXXu.*model.Knm ))/exp(2*model.GP.logtheta(1)); 
          else
             % derivatives over the kernel quantities K_mm, K_nm and
             % tr(K_nn) wrt kernel variance  sigmaf^2 = exp(2*log(sigmaf)) 
             if strcmp(model.objectFunc, 'var') 
                DTrKnn(d) = 2*model.TrKnn; 
             end
             % derivative of the hyperparameter
             Dhypmm(d) = 2*sum(sum( model.Kmm));
             Dhypnm(d) = 2*sum(sum( model.Knm)); 
          end     
       end
    
    case 'seard' 
       for d=1:model.GP.nParams   
          if d <= model.D 
             %model.Xu(:,d) = model.Xu(:,d)/exp(model.GP.logtheta(d));
             %model.X(:,d) = model.X(:,d)/exp(model.GP.logtheta(d));
             % derivatives over the kernel quantities K_mm, K_nm and
             % tr(K_nn) wrt lengthscale
             % ell_d^2 = exp(2*log(ell_d)) and the deriv. is wrt log(ell_d)
             DKmm = ( (  model.Xu(:,d)*ones(1, model.m) - ones(model.m,1)*(model.Xu(:,d)') ).^2 ).*model.Kmm;
             DKnm = ( (  model.X(:,d)*ones(1, model.m) - ones(model.n,1)*(model.Xu(:,d)') ).^2  ).*model.Knm;
           
             % derivative of the hyperparameters
             Dhypmm(d) = sum(sum( DKmm.*Tmm ))/exp(2*model.GP.logtheta(d));
             Dhypnm(d) = sum(sum( DKnm.*Tnm ))/exp(2*model.GP.logtheta(d));
          else
             % derivatives over the kernel quantities K_mm, K_nm and
             % tr(K_nn) wrt kernel variance  sigmaf^2 = exp(2*log(sigmaf))
             if strcmp(model.objectFunc, 'var') 
                DTrKnn(d) = 2*model.TrKnn; 
             end
             % derivative of the hyperparameter
             Dhypmm(d) = 2*sum(sum( model.Kmm.*Tmm ));
             Dhypnm(d) = 2*sum(sum( model.Knm.*Tnm ));
          end
       %         
       end
    end
%    
elseif strcmp(model.indType, 'weights')
%
    switch model.GP.type 
    case 'se'
       for d=1:model.GP.nParams
          DKmm1 = zeros(model.m, model.m);
          DKnm1 = zeros(model.n, model.m); 
          if d == 1 
          % derivatives over the kernel quantities K_mm, K_nm and
          % tr(K_nn) wrt the shared lengthscale
          % ell^2 = exp(2*log(ell)) and the deriv. is wrt log(ell)    
          cnt = 0;
          for j=1:model.nIndParams
             DKmm = model.X(model.IndSubset(:,j),:)*model.X(model.IndSubset(:,j),:)';
             dgXmm = diag(DKmm);
             DKmm = dgXmm*ones(1, model.m) + ones(model.m,1)*(dgXmm') - 2*DKmm;
             DKmm = (1/exp(2*model.GP.logtheta(1)))*(DKmm.*model.KmmSq(:,:,j));
             %W = sparse(diag(model.Xu(:,j)));
             W = model.W(:,j)*ones(1, model.m);
             %DKmm1 = DKmm1 + W*(DKmm*W);
             DKmm1 = DKmm1 + W.*(DKmm.*W');
            
             % cross-covariance matrix between training inputs and inducing variables
             DKnm = -2*model.X*model.X(model.IndSubset(:,j),:)' + ...
                    (sum(model.X.*model.X, 2))*ones(1, model.m) + ...
                    ones(model.n,1)*(sum(model.X(model.IndSubset(:,j),:).*model.X(model.IndSubset(:,j),:),2)');
             DKnm = (1/exp(2*model.GP.logtheta(1)))*(DKnm.*model.KnmAll(:,:,j)); 
             DKnm1 = DKnm1 + DKnm.*( ones(model.n, 1)*model.W(:,j)' );  
             %
             for k=j+1:model.nIndParams
                 cnt = cnt + 1; 
                 DKmm = -2*model.X(model.IndSubset(:,j),:)*model.X(model.IndSubset(:,k),:)'...
                      + (sum(model.X(model.IndSubset(:,j),:).*model.X(model.IndSubset(:,j),:),2))*ones(1, model.m)...
                      + ones(model.m,1)*(sum(model.X(model.IndSubset(:,k),:).*model.X(model.IndSubset(:,k),:),2)');
                 DKmm = (1/exp(2*model.GP.logtheta(1)))*(DKmm.*model.KmmCr(:,:,cnt));
                 DKmm = W.*(DKmm.*(ones(model.m, 1)*model.W(:,k)') );
                 DKmm1 = DKmm1 + (DKmm + DKmm'); 
             end
          end
   
          else
             % derivatives over the kernel quantities K_mm, K_nm and
             % tr(K_nn) wrt kernel variance  sigmaf^2 = exp(2*log(sigmaf)) and the deriv. is wrt log(ell)
             DKmm1 = 2*model.Kmm;
             DKnm1 = 2*model.Knm;
             if strcmp(model.objectFunc, 'var') 
                DTrKnn(d) = 2*model.TrKnn; 
             end
          end
          
          % derivative of the hyperparameters
          Dhypmm(d) = sum(sum( DKmm1.*Tmm ));
          Dhypnm(d) = sum(sum( DKnm1.*Tnm ));
          
       end
    case 'seard' 
       for d=1:model.GP.nParams   
          DKmm1 = zeros(model.m, model.m);
          DKnm1 = zeros(model.n, model.m);
          if d <= model.D 
             model.X(:,d) = model.X(:,d)/exp(model.GP.logtheta(d));
             % derivatives over the kernel quantities K_mm, K_nm and
             % tr(K_nn) wrt lengthscale
             % ell_d^2 = exp(2*log(ell_d)) and the deriv. is wrt log(ell_d)
             cnt = 0;
             for j=1:model.nIndParams
                 XX = model.X(model.IndSubset(:,j),d);
                 DKmm = ( (  XX*ones(1, model.m) - ones(model.m,1)*(XX') ).^2 ).*model.KmmSq(:,:,j);
                 %W = sparse(diag(model.Xu(:,j)));
                 W = model.Xu(:,j)*ones(1, model.m);
                 %DKmm1 = DKmm1 + W*(DKmm*W);
                 DKmm1 = DKmm1 + W.*(DKmm.*W');
                 % cross-covariance matrix between training inputs and inducing variables
                 DKnm = ( (  model.X(:,d)*ones(1, model.m) - ones(model.n,1)*(XX') ).^2  ).*model.KnmAll(:,:,j);
                 %DKnm1 = DKnm1 + DKnm*W;
                 DKnm1 = DKnm1 + DKnm.*( ones(model.n, 1)*model.Xu(:,j)' );  
                 for k=j+1:model.nIndParams
                    cnt = cnt + 1; 
                    DKmm = ((  model.X(model.IndSubset(:,j),d)*ones(1, model.m) ...
                         - ones(model.m,1)*(model.X(model.IndSubset(:,k),d)' )).^2 ).*model.KmmCr(:,:,cnt);
                    %DKmm = sparse(diag( model.Xu(:,j)))*(DKmm*sparse(diag(model.Xu(:,k)))); 
                    DKmm = W.*(DKmm.*(ones(model.m, 1)*model.Xu(:,k)') );
                    DKmm1 = DKmm1 + (DKmm + DKmm');
                 end
             end
             
          else
             % derivatives over the kernel quantities K_mm, K_nm and
             % tr(K_nn) wrt kernel variance  sigmaf^2 = exp(2*log(sigmaf)) and the deriv. is wrt log(ell)
             DKmm1 = 2*model.Kmm;
             DKnm1 = 2*model.Knm;
             if strcmp(model.objectFunc, 'var') 
                DTrKnn(d) = 2*model.TrKnn; 
             end
          end

          % derivative of the hyperparameters
          Dhypmm(d) = sum(sum( DKmm1.*Tmm ));
          Dhypnm(d) = sum(sum( DKnm1.*Tnm ));
       %         
       end
    end 
end