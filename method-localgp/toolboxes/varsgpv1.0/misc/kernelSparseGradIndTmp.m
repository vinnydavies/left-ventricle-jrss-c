function DXu = kernelSparseGradInd(model, Tmm, Tnm)
%
%

DXu = zeros(model.nIndParams, model.m);
switch model.GP.type 
  case 'se' 
   for d = 1:model.nIndParams
   %    
      DKnm = (( model.X(:,d)*ones(1,model.m) - ones(model.n,1)*(model.Xu(:,d)') )/exp(2*model.GP.logtheta(1))).*model.Knm;
      DKmm = -(( ones(model.m,1)*(model.Xu(:,d)') - model.Xu(:,d)*ones(1,model.m) )/exp(2*model.GP.logtheta(1))).*model.Kmm;       
    
      DXu(d,:) = sum( DKmm.*Tmm , 1 ) + sum( DKnm.*Tnm, 1)/model.sigma2;
   %
   end  
  case 'seard' 
   for d = 1:model.nIndParams
   %    
      DKnm = (( model.X(:,d)*ones(1,model.m) - ones(model.n,1)*(model.Xu(:,d)') )/exp(2*model.GP.logtheta(d))).*model.Knm;
      DKmm = -(( ones(model.m,1)*(model.Xu(:,d)') - model.Xu(:,d)*ones(1,model.m) )/exp(2*model.GP.logtheta(d))).*model.Kmm;       
    
      DXu(d,:) = sum( DKmm.*Tmm , 1 ) + sum( DKnm.*Tnm, 1)/model.sigma2;
   %
   end
end

