function [DXumm DXunm] = kernelSparseGradInd(model, Tmm, Tnm)
%
%

DXumm = zeros(model.nIndParams, model.m); 
DXunm = zeros(model.nIndParams, model.m);
    
if strcmp(model.indType, 'pseudoIns')
   switch model.GP.type 
   case 'se' 
       if strcmp(model.indRepar, 'no')        
          for d = 1:model.D
          %    
              DKnm = ( model.X(:,d)*ones(1,model.m) - ones(model.n,1)*(model.Xu(:,d)')  ).*model.Knm;
              DKmm = -( ones(model.m,1)*(model.Xu(:,d)') - model.Xu(:,d)*ones(1,model.m) ).*model.Kmm;       
  
              DXumm(d,:) = sum( DKmm.*Tmm, 1)/exp(2*model.GP.logtheta(1));  
              DXunm(d,:) = sum( DKnm.*Tnm, 1)/exp(2*model.GP.logtheta(1));
          %
          end
       else        
          model.Knm = model.Knm.*Tnm;
          model.Kmm = model.Kmm.*Tmm;
          for r = 1:model.R
          %    
              preFact0 =  model.Xu*model.X(model.IndSubset(:,r),:)';
              preFact = diag(preFact0)';
              
              if strcmp(model.precomp, 'yes')
                  %Tnm = model.XXr(:,:,r) - ones(model.n,1)*preFact;
                  Tnm = bsxfun(@minus, model.XXr(:,:,r), preFact);
              else
                  %Tnm = model.X*model.X(model.IndSubset(:,r),:)' - ones(model.n,1)*preFact;
                  Tnm = bsxfun(@minus, model.X*model.X(model.IndSubset(:,r),:)', preFact);
              end
              %Tmm =  preFact0 - ones(model.m,1)*preFact;
              Tmm =  bsxfun(@minus, preFact0,  preFact); 
              
              DXumm(r,:) = sum( Tmm.*model.Kmm, 1)/exp(2*model.GP.logtheta(1));  
              DXunm(r,:) = sum( Tnm.*model.Knm, 1)/exp(2*model.GP.logtheta(1));
          %   
          end 
       end
       case 'seard' 
       if strcmp(model.indRepar, 'no')   
          for d = 1:model.nIndParams
          %    
              DKnm = (( model.X(:,d)*ones(1,model.m) - ones(model.n,1)*(model.Xu(:,d)') )/exp(2*model.GP.logtheta(d))).*model.Knm;
              DKmm = -(( ones(model.m,1)*(model.Xu(:,d)') - model.Xu(:,d)*ones(1,model.m) )/exp(2*model.GP.logtheta(d))).*model.Kmm;       
    
              DXumm(d,:) = sum( DKmm.*Tmm, 1);  
              DXunm(d,:) = sum( DKnm.*Tnm, 1);
          %
          end
       else
          model.X =  model.X./ ( ones(model.n,1)*(exp(model.GP.logtheta(1:model.D))'));
          model.Xu =  model.Xu./ ( ones(model.m,1)*(exp(model.GP.logtheta(1:model.D))'));
          for r = 1:model.R
          %    
              preFact = sum(model.Xu.*model.X(model.IndSubset(:,r),:),2)';
              DKnm = ( model.X*model.X(model.IndSubset(:,r),:)' - ones(model.n,1)*preFact ).*model.Knm;
              DKmm = -( ones(model.m,1)* preFact - model.Xu*model.X(model.IndSubset(:,r),:)').*model.Kmm;       
    
              DXumm(r,:) = sum( DKmm.*Tmm, 1);  
              DXunm(r,:) = sum( DKnm.*Tnm, 1);
          %
          end
       end
   end
%   
elseif strcmp(model.indType, 'weights')
%
   for d=1:model.nIndParams
   %    
       DKnm = model.KnmAll(:,:,d);
       W = ones(model.m, 1)*model.W(:,d)';
       DKmm = model.KmmSq(:,:,d).*W;
       if d==1   
          %W = sparse(diag(model.Xu(:,2)));
          %G = model.KmmCr(:,:,1)*W;
          G = model.KmmCr(:,:,1).*( ones(model.m, 1)*model.W(:,2)' );
          DKmm = DKmm + G;
       elseif d==2
          G = ( model.W(:,1)*ones(1, model.m) ).*model.KmmCr(:,:,1); 
          DKmm = DKmm + G';
       end
       DXumm(d,:) = sum( DKmm.*Tmm, 2)';
       DXunm(d,:) = sum( DKnm.*Tnm, 1);
   %
   end
end