function model = returnOptimizedParams(model, W, flag)
%

% optimize over both inducing variable parameters and model 
% (kernel and likelihood) hyperprameters  
if flag == 1
    % inducing variables parameters
    en = model.m*model.nIndParams;
    if strcmp(model.indType, 'weights')
         model.W = reshape(W(1:en), model.m,  model.nIndParams);    
    elseif strcmp(model.indType, 'pseudoIns')
         if strcmp(model.indRepar, 'yes')  
             model.W = reshape(W(1:en), model.m,  model.nIndParams);   
             Xu = zeros(model.m, model.D);
             for r=1:model.R
                  Xu = Xu + model.X(model.IndSubset(:,r),:).*repmat(model.W(:,r),1,model.D);
             end 
             model.Xu = Xu;
         else
             model.Xu = reshape(W(1:en), model.m,  model.nIndParams);    
         end
    end
    % kernel hyperparameters
    st = en + 1;   
    en = en + model.GP.nParams;
    model.GP.logtheta = W(st:en); 
    % likelihood hyperparameters
    st = en + 1; 
    en = en + model.Likelihood.nParams;
    if strcmp(model.Likelihood.type, 'Gaussian') == 1
        model.Likelihood.logtheta = W(st:en); 
    end
% optimize over only inducing variable parameters   
elseif flag == 2
    % inducing variables parameters
    en = model.m*model.nIndParams;
    if strcmp(model.indType, 'weights')
         model.W = reshape(W(1:en), model.m,  model.nIndParams);    
    elseif strcmp(model.indType, 'pseudoIns')
         if strcmp(model.indRepar, 'yes')  
             model.W = reshape(W(1:en), model.m,  model.nIndParams);   
             Xu = zeros(model.m, model.D);
             for r=1:model.R
                  Xu = Xu + model.X(model.IndSubset(:,r),:).*repmat(model.W(:,r),1,model.D);
             end 
             model.Xu = Xu;
         else
             model.Xu = reshape(W(1:en), model.m,  model.nIndParams);    
         end
    end
% optimize over only model (kernel and likelihood) hyperparameters   
elseif flag == 3
    % kernel hyperparameters
    st = 1;   
    en = model.GP.nParams;
    model.GP.logtheta = W(st:en); 
    % extract likelihood hyperparameters
    st = en + 1; 
    en = en + model.Likelihood.nParams;
    if strcmp(model.Likelihood.type, 'Gaussian') == 1
        model.Likelihood.logtheta = W(st:en); 
    end
end
model.sigma2 = exp(2*model.Likelihood.logtheta);

