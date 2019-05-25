function W = extractOptimizedParams(model, flag)
%
%

if strcmp(model.indType, 'weights')
    Xu = model.W;
elseif strcmp(model.indType, 'pseudoIns')     
    if strcmp(model.indRepar, 'yes') 
       Xu = model.W;
    else
       Xu = model.Xu;
    end
end
    
% place optimized parameters in a single vector
switch model.Likelihood.type 
    case 'Gaussian' 
       if flag == 1
          W = [reshape(Xu, model.m*model.nIndParams, 1); model.GP.logtheta; model.Likelihood.logtheta];
       elseif flag == 2
          W = [reshape(Xu, model.m*model.nIndParams, 1)];
       elseif flag == 3
          W = [model.GP.logtheta; model.Likelihood.logtheta];
       end
    case 'Probit'
       if flag == 1
          W = [reshape(Xu, model.m*model.nIndParams, 1); model.GP.logtheta];
       elseif flag == 2
          W = [reshape(Xu, model.m*model.nIndParams, 1)];
       elseif flag == 3
          W = [model.GP.logtheta];
       end       
end


