function model = varsgpCreate(kern, X, y, options)
%function model = varsgpCreate(kern, X, y, options)
%
%
%

model.type = 'gpmodel';
if isfield(options, 'objectFunc') 
    model.objectFunc = options.objectFunc;
else
    model.objectFunc = 'var';
end
    
model.Likelihood.type = options.Likelihood;
model.GP.type = kern;

[n D] = size(X);
model.n = n;
model.D = D;
model.y = y(:); 
model.X = X;

% initialize parameters of the inducing variables
if strcmp(options.indType, 'pseudoIns')
%   
    model.m = options.m; 
    model.indType = options.indType;
    % number of inducing variables
    
    if ~isfield(options, 'indRepar')
         options.indRepar = 'no';
    end
    
    model.indRepar = options.indRepar;
    if strcmp(model.indRepar, 'no')        
        % randomly initialize pseudo inputs from the training inputs
        perm = randperm(n);
        model.Xu = X(perm(1:model.m),:);     
        model.Xuinit = model.Xu;
        model.nIndParams = size(model.Xu, 2);
    else
        if options.R*model.m > n
           options.R = floor(n/model.m);
        end
        model.R = options.R;
   
        IndSubset = zeros(model.m,model.R);
        % initialize inducing set based on k-means
        perm = randperm(size(X,1));
        ops(1) = 1;
        ops(14)  = 10;
        centres = kmeans(X(perm(1:model.m),:), X, ops);
        d2 = dist2(X, centres);
        % for each center find the R nearest neighbors 
        for m=1:model.m
            [sortvals, index]  = sort(d2(:,m));
            IndSubset(m,:) = index(1:options.R); 
        end
        model.IndSubset = IndSubset;
        
        % if we have a binary classification problem, then choose half inducing variables 
        % for each class  
        %if strcmp(model.Likelihood.type, 'Probit')
        %    PosClass =  find(y==1);
        %    NegClass = find(y==-1);
        %    perm = randperm(length(PosClass));
        %    for m=1:floor(model.m/2) 
        %         if m*model.R >  length(PosClass)
        %             m = m -1;
        %             break;
        %         else
        %            model.IndSubset(m,:) = PosClass(perm((m-1)*model.R+1:m*model.R));
        %         end
        %    end
        %    perm = randperm(length(NegClass));
        %    for m1=m+1:model.m 
        %         if m1*model.R > length(NegClass)
        %             break;
        %         else
        %             model.IndSubset(m1,:) = NegClass(perm((m1-1)*model.R+1:m1*model.R));
        %         end
        %    end
        %else
        %    % regression case all inducing inputs are treated equally
        %    perm = randperm(n);
        %    for r=1:model.R
        %        model.IndSubset(:,r) = perm((r-1)*model.m+1:r*model.m);
        %   end
        %end
        
        model.precomp = options.precomp; 
        if strcmp(model.precomp, 'yes')
            model.XXr = zeros(n, model.m, model.R);    
            for r=1:model.R 
               model.XXr(:,:,r) = model.X*model.X(model.IndSubset(:,r),:)';
            end
        end
         
        % initialize randomly the linear weights 
        model.W = 0.01*randn(model.m, model.R) + 1/model.R;
        model.Winit = model.W; 
        % number of inducing variable parameter (per inducing variable) 
        model.nIndParams = size(model.W, 2);
    end
%    
elseif strcmp(options.indType, 'weights')
%    
    if (2*options.m <= n)
      model.indType = options.indType;
      % number of inducing variables
      model.m = options.m; 
      % choose two random sets of training points of size m
      perm = randperm(n);
      model.IndSubset(:,1) = perm(1:model.m);
      model.IndSubset(:,2) = perm(model.m+1:2*model.m);
      
      % initialize randomly the linear weights 
      model.W = 0.5 + 0.1*randn(model.m, 2);
      model.Winit = model.W;
      % number of inducing variable parameter (per inducing variable) 
      model.nIndParams = size(model.W, 2);
    else
      error('The number of inducing variables when the indType=weigths must be less than a half of the number of training examples');
    end
%
else
    error('Unknown inducing variable type');
end

% number of inducing variables
model.m = options.m; 
jitter = 1e-7; 
switch model.Likelihood.type
%    
    case 'Gaussian' % standard regression
         %
         model.Likelihood.nParams = 1; % parameters (excluding gp function F)
         % log(sigma) parameter,  i.e the nosie variance sigma2 = exp(2*log(sigma))  
         model.Likelihood.logtheta = 0.5*log(var(y, 1)/4);
         % precompute also the dot product of the outputs
         model.yy= y'*y;
         model.vary = var(y);
         model.jitter = jitter*model.vary;
         %
    case 'Probit'  % binary classifcation
         %
         model.Likelihood.nParams = 1;
         model.Likelihood.logtheta = 0; 
         model.jitter = jitter;
         %
    case 'Poisson' % for counts data      
         %
         % !!! not implemented yet !!!
         model.Likelihood.nParams = 0;  
%         
end     

%
switch kern
    case 'se'
       model.GP.type = 'se';
       % kernel hyperparameters
       dd = log((max(X)-min(X))'/2 + eps);
       dd(dd==-Inf)=0;
       model.GP.logtheta(1,1) = mean(dd);
       if strcmp('model.Likelihood.type','Gaussian')
          model.GP.logtheta(2,1) = 0.5*log(var(y,1));
       else
          model.GP.logtheta(2,1) = 0;
       end
       model.GP.nParams = size(model.GP.logtheta,1);
       model.GP.constDiag = 1; 
    %   
    case 'seard'
       model.GP.type = 'seard';
       % kernel hyperparameters
       dd = log((max(X)-min(X))'/2);
       dd(dd==-Inf)=0;
       model.GP.logtheta(1:D,1) = dd;
       if strcmp('model.Likelihood.type','Gaussian')
          model.GP.logtheta(D+1,1) = 0.5*log(var(y,1));
       else
          model.GP.logtheta(D+1,1) = 0;
       end
       model.GP.nParams = size(model.GP.logtheta,1);
       model.GP.constDiag = 1; 
end
