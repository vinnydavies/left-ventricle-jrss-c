function runDataset(dataSet, hyperparam, indparam, Repititions)
%
%

warning off;
Iterations = 500;
typeKL = 'diag';
outdir = 'results/';
dataSet = char(dataSet);

% load dataset
switch dataSet
   case 'Abalone'
      load datasets/abalone_split.mat;  
      M = [16 32 64 128 256 512 1024]; 
   case 'Boston' 
      % run all sparse algorithms in Boston Housing data
      load  datasets/data_boston.mat;
      Xtrain = X; Ytrain = y; Xtest = Xstar; Ytest = ystar;      
      % choices of the number of pseudo inputs 
      M = [5 10 20, 30:30:455, 455];
   case 'Pumadyn32nm'
      load datasets/pumadyn32nm.mat;
      Ytrain = T_tr; T_tr = []; 
      Ytest = T_tst; T_tst = [];  
      Xtrain = X_tr; X_tr = [];   
      Xtest = X_tst; X_tst = [];
      M = [16 32 64 128 256 512 1024];
   case 'Pendulum'
      load datasets/pendulum.mat;
      Ytrain = T_tr; T_tr = []; 
      Ytest = T_tst; T_tst = [];  
      Xtrain = X_tr; X_tr = [];   
      Xtest = X_tst; X_tst = [];
      M = [5 10 20, 30:30:300, 315];   
    case 'Pol'
      load datasets/pol.mat;
      Ytrain = T_tr; T_tr = []; 
      Ytest = T_tst; T_tst = [];  
      Xtrain = X_tr; X_tr = [];   
      Xtest = X_tst; X_tst = [];
      M = [16 32 64 128 256 512 1024];  
    case 'Elevators'
      load datasets/elevators.mat;
      Ytrain = T_tr; T_tr = []; 
      Ytest = T_tst; T_tst = [];  
      Xtrain = X_tr; X_tr = [];   
      Xtest = X_tst; X_tst = [];
      M = [16 32 64 128 256 512 1024];
    case 'Kin40k'
      load datasets/kin40k.mat;
      Ytrain = T_tr; T_tr = []; 
      Ytest = T_tst; T_tst = [];  
      Xtrain = X_tr; X_tr = [];   
      Xtest = X_tst; X_tst = [];
      M = [16 32 64 128 256 512 1024];  
end

[Ntrain D] = size(Xtrain);
Ntest = size(Xtest,1);


% do not normalize for the Boston housing, Kin40k and Abalone data 
% becuase there are rouhgly normalized 
if (~strcmp(dataSet, 'Boston') &  ~strcmp(dataSet, 'Kin40k')  &  ~strcmp(dataSet, 'Abalone'))
   disp(dataSet); 
   % normalize the data to the inputs have unit variacen and zero mean 
   % and the ouput zero mean 
   meanX = mean(Xtrain);
   sqrtvarX = sqrt(var(Xtrain)); 
   meanY = mean(Ytrain); 
   Ytrain = Ytrain - meanY;
   Ytest = Ytest - meanY; 
   Xtrain = Xtrain - repmat(meanX, Ntrain, 1);
   Xtrain = Xtrain./repmat(sqrtvarX, Ntrain, 1);
   Xtest = Xtest - repmat(meanX, Ntest,1);
   Xtest = Xtest./repmat(sqrtvarX, Ntest, 1);
end

% initialization of the model hyperparameters 
logtheta0(1:D,1) = log((max(Xtrain)-min(Xtrain))'/2);
logtheta0(D+1,1) = 0.5*log(var(Ytrain,1)); 
logtheta0(D+2,1) = 0.5*log(var(Ytrain,1)/4);   


% for the pumadyn32nm initialize based on a subset of the training data 
if strcmp(dataSet, 'Pumadyn32nm') 
    covfunc = {'covSum', {'covSEard','covNoise'}};
    perm = randperm(Ntrain); 
    [logtheta0 fw] = minimize(logtheta0, 'gpr', -200, covfunc, Xtrain(perm(1:1024), :), Ytrain(perm(1:1024)) );
end
    

if Ntrain <= 8000
   IterationsFull = 500;
else
   % for very larse datasets run the Full GP for fewer iterations 
   IterationsFull = 200;
end


% RUN FULL GP using R&W software
if ~exist([outdir 'FullGP' dataSet '.mat'], 'file'),
    covfunc = {'covSum', {'covSEard','covNoise'}};
    tic;
    [fulllogtheta fw] = minimize(logtheta0, 'gpr', -IterationsFull, covfunc, Xtrain, Ytrain);
    FullGP.elapsedTrain = toc;
   
    if strcmp(typeKL, 'Full') 
    [v, K] = feval(covfunc{:}, fulllogtheta, Xtrain, Xtrain);
    [v, Ktest] = feval(covfunc{:}, fulllogtheta, Xtest, Xtest);
    [v, a] = feval(covfunc{:}, fulllogtheta, Xtrain, Xtest);
    sigma2n = exp(2*fulllogtheta(end));
    L = chol(K+sigma2n*eye(Ntrain))';
    invK = L'\(L\eye(Ntrain));
    Full_mutest = a'*(invK*Ytrain);
    Full_Ktest = Ktest - a'*(invK*a);  
    Full_KtestL = jitterChol(Full_Ktest)';       
    end
    
    % error prediction measure;
    tic;
    [Fullmu, Fulls2] = gpr(fulllogtheta, covfunc, Xtrain, Ytrain, Xtest);
    FullGP.elapsedPred = toc;
     
    FullGP.mse = errorMeasureRegress(Fullmu, Fulls2, Ytest, 'mse');   
    FullGP.nlp = errorMeasureRegress(Fullmu, Fulls2, Ytest, 'nlp');
    FullGP.smse = errorMeasureRegress(Fullmu, Fulls2, Ytest, 'smse');
    FullGP.snlp = errorMeasureRegress(Fullmu, Fulls2, Ytest, 'snlp', Ytrain);
    FullGP.logL = -fw(end);   
    FullGP.params = fulllogtheta';
    FullGP.iters = size(fw(:),1);
    % save results 
    save([outdir 'FullGP' dataSet '.mat'], 'FullGP', 'fulllogtheta');
else
    fprintf('File exists not need to rerun it...\n'); 
    load([outdir 'FullGP' dataSet '.mat']);
    covfunc = {'covSum', {'covSEard','covNoise'}}; 
    [Fullmu, Fulls2] = gpr(fulllogtheta, covfunc, Xtrain, Ytrain, Xtest);
    if strcmp(typeKL, 'Full') 
    [v, K] = feval(covfunc{:}, fulllogtheta, Xtrain, Xtrain);
    [v, Ktest] = feval(covfunc{:}, fulllogtheta, Xtest, Xtest);
    [v, a] = feval(covfunc{:}, fulllogtheta, Xtrain, Xtest);
    sigma2n = exp(2*fulllogtheta(end));
    L = chol(K+sigma2n*eye(Ntrain))';
    invK = L'\(L\eye(Ntrain));
    Full_mutest = a'*(invK*Ytrain);
    Full_Ktest = Ktest - a'*(invK*a);   
    Full_KtestL = jitterChol(Full_Ktest)';  
    end
    
end

% Fix seeds
randn('seed', 1e6);
rand('seed', 1e6);
perm = zeros(Repititions, Ntrain);
for it = 1:Repititions
  perm(it,:) = randperm(Ntrain); 
end

% RUN THE SPARSE METHODS
% 
for it = 1:Repititions
%    
  % set them to those of full GP (or large subset of data) if there are fixed 
  if strcmp(hyperparam, 'fixed')
    logtheta0 = fulllogtheta;
  end      
  % run for different number of inducing variables
  for i=1:size(M,2)
    %  
    m = M(i);
    fprintf('Datasets:%s, Hyperpar:%s, Ind-inputs:%s, Run:%d/%d, #Ind-vars:%d\n', dataSet, hyperparam, indparam, it, Repititions, m);
    
    % initial values for the pseudo inputs 
    Xuinit = Xtrain(perm(it,1:m),:);
    Yuinit = Ytrain(perm(it,1:m),:);
   
    %
    % RUN THE SUBSET OF DATA METHOD 
    if 0% strcmp(hyperparam, 'free') 
       covfunc = {'covSum', {'covSEard','covNoise'}};
       tic;
       [logtheta fw] = minimize(logtheta0, 'gpr', -Iterations, covfunc, Xuinit, Yuinit);
       SD.elapsedTrain(it,i) = toc;
    else
       logtheta = logtheta0;
    end
    if strcmp(typeKL, 'Full') 
    [v, Kuu] = feval(covfunc{:}, fulllogtheta, Xuinit, Xuinit);
    sigma2n = exp(2*fulllogtheta(end));
    Lu = chol(Kuu+sigma2n*eye(m))';
    Sigma = Lu'\(Lu\eye(m));
    [v, KKnu] = feval(covfunc{:}, fulllogtheta, Xtest, Xuinit);
    [v, Ktest] = feval(covfunc{:}, fulllogtheta, Xtest, Xtest);
    mu = KKnu*(Sigma*Yuinit);
    Ktest = Ktest - KKnu*Sigma*KKnu'; 
    KtestL = jitterChol(Ktest)'; 
    %SD.KLpq(it,i) = KL(Full_mutest, Full_Ktest, mu, K);
    %SD.KLqp(it,i)  = KL(mu, K, Full_mutest, Full_Ktest);  
    SD.KLpq(it,i) = KLchol(Full_mutest, Full_KtestL, mu, KtestL);
    SD.KLqp(it,i) = KLchol(mu, KtestL, Full_mutest, Full_KtestL); 
    end
    
    % take the prediction in the test set 
    tic;
    [mu, s2] = gpr(logtheta, covfunc, Xuinit, Yuinit, Xtest);
    SD.elapsedPred(it,i)=toc;
      
    res = mu-Ytest;
    SD.mse(it,i) = sum(res.^2)/Ntest;
    SD.nlp(it,i) = 0.5*mean(log(2*pi*s2)+res.^2./s2); 
    SD.smse(it,i) = SD.mse(it,i)/var(Ytest,1);
    muYtrain = mean(Ytrain);   varYtrain = var(Ytrain,1);
    SD.snlp(it,i) = SD.nlp(it,i) - 0.5*mean(log(2*pi*varYtrain) + ((Ytest-muYtrain).^2)/varYtrain);
    SD.logL(it,i) = -gpr(logtheta, covfunc, Xuinit, Yuinit);  
    SD.KLpqDiag(it,i) = KLdiag(Fullmu, Fulls2, mu, s2);
    SD.KLqpDiag(it,i) = KLdiag(mu, s2, Fullmu, Fulls2);
    SD.params(:, it,i) = logtheta;
    
    
    % RUN THE VARIATIONAL METHOD 
    clear options; 
    options.Likelihood = 'Gaussian';
    options.m = m; 
    options.indType = 'pseudoIns';
    options.objectFunc = 'var';
    model = varsgpCreate('seard', Xtrain, Ytrain, options);
    model.GP.logtheta = logtheta0(1:end-1);
    model.Likelihood.logtheta = logtheta0(end);
    model.Xu = Xuinit;
    trops(1) = Iterations; % number of iterations
    trops(2) = 1; 
    trops(3) = 1;
    % check if you need optimization before prediction
    if strcmp(hyperparam, 'free') | strcmp(indparam, 'free')
       if strcmp(hyperparam, 'free') & strcmp(indparam, 'fixed')
           trops(3) = 3;
       end
       if strcmp(hyperparam, 'fixed') & strcmp(indparam, 'free')
           trops(3) = 2;
       end
       tic; 
       [model margLogL] = varsgpTrain(model, trops);
       VAR.elapsedTrain(it,i)=toc;
       tic; 
    end
    tic;
    [mutest, vartest] = varsgpPredict(model, Xtest);
    vartest = vartest + exp(2*model.Likelihood.logtheta);
    VAR.elapsedPred(it,i)=toc;
    VAR.mse(it,i)  = errorMeasureRegress(mutest, vartest, Ytest, 'mse');
    VAR.nlp(it,i)  = errorMeasureRegress(mutest, vartest, Ytest, 'nlp');
    VAR.smse(it,i) = errorMeasureRegress(mutest, vartest, Ytest, 'smse');
    VAR.snlp(it,i) = errorMeasureRegress(mutest, vartest, Ytest, 'snlp', Ytrain);
    W = [reshape(model.Xu, model.m*model.nIndParams, 1); model.GP.logtheta; model.Likelihood.logtheta];
    VAR.logL(it,i) = - varsgpLikelihood(W, model, 1);  
    
    if strcmp(typeKL, 'Full') 
    [mj, vj, Ktest] = varsgpPredict(model, Xtest);
    KtestL = jitterChol(Ktest)'; 
    %VAR.KLpq(it,i) = KL(Full_mutest, Full_Ktest, mutest, Ktest);
    %VAR.KLqp(it,i) = KL(mutest, Ktest, Full_mutest, Full_Ktest);
    VAR.KLpq(it,i) = KLchol(Full_mutest, Full_KtestL, mutest, KtestL);
    VAR.KLqp(it,i) = KLchol(mutest, KtestL, Full_mutest, Full_KtestL);
    end
    
    VAR.KLpqDiag(it,i) = KLdiag(Fullmu, Fulls2, mutest, vartest);
    VAR.KLqpDiag(it,i) = KLdiag(mutest, vartest, Fullmu, Fulls2);
    VAR.params(:,it,i) = [model.GP.logtheta', model.Likelihood.logtheta];
    
    
    % RUN THE DTC METHOD
    clear options; 
    options.Likelihood = 'Gaussian';
    options.m = m; 
    options.indType = 'pseudoIns';
    options.objectFunc = 'dtc';
    model = varsgpCreate('seard', Xtrain, Ytrain, options);
    model.GP.logtheta = logtheta0(1:end-1);
    model.Likelihood.logtheta = logtheta0(end);
    model.Xu = Xuinit;
    trops(1) = Iterations; % number of iterations
    trops(2) = 1; 
    trops(3) = 1;
    % check if you need optimization before prediction
    if strcmp(hyperparam, 'free') | strcmp(indparam, 'free')
       if strcmp(hyperparam, 'free') & strcmp(indparam, 'fixed')
           trops(3) = 3;
       end
       if strcmp(hyperparam, 'fixed') & strcmp(indparam, 'free')
           trops(3) = 2;
       end
       tic;
       [model margLogL] = varsgpTrain(model, trops);
       DTC.elapsedTrain(it,i)=toc;
       tic;  
    end
    % prediction
    tic;  
    [mutest, vartest] = varsgpPredict(model, Xtest);
    vartest = vartest + exp(2*model.Likelihood.logtheta);
    DTC.elapsedPred(it,i)=toc;
    DTC.mse(it,i)  = errorMeasureRegress(mutest, vartest, Ytest, 'mse');
    DTC.nlp(it,i)  = errorMeasureRegress(mutest, vartest, Ytest, 'nlp');
    DTC.smse(it,i) = errorMeasureRegress(mutest, vartest, Ytest, 'smse');
    DTC.snlp(it,i) = errorMeasureRegress(mutest, vartest, Ytest, 'snlp', Ytrain);
    W = [reshape(model.Xu, model.m*model.nIndParams, 1); model.GP.logtheta; model.Likelihood.logtheta];
    DTC.logL(it,i) = -dtcsgpLikelihood(W, model, 1); 
    
    if strcmp(typeKL, 'Full')
    [mj, vj, Ktest] = varsgpPredict(model, Xtest);
    KtestL = jitterChol(Ktest)'; 
    %DTC.KLpq(it,i) = KL(Full_mutest, Full_Ktest, mutest, Ktest);
    %DTC.KLqp(it,i) = KL(mutest, Ktest, Full_mutest, Full_Ktest);
    DTC.KLpq(it,i) = KLchol(Full_mutest, Full_KtestL, mutest, KtestL);
    DTC.KLqp(it,i) = KLchol(mutest, KtestL, Full_mutest, Full_KtestL);
    end
    
    DTC.KLpqDiag(it,i) = KLdiag(Fullmu, Fulls2, mutest, vartest);
    DTC.KLqpDiag(it,i) = KLdiag(mutest, vartest, Fullmu, Fulls2);
    DTC.params(:,it,i) = [model.GP.logtheta', model.Likelihood.logtheta];   
    %
    
    % RUN Ed Snelson's code for FITC
    if strcmp(hyperparam, 'free') | strcmp(indparam, 'free')
       if strcmp(hyperparam, 'free') & strcmp(indparam, 'fixed')
           trops(3) = 3;
       end
       if strcmp(hyperparam, 'fixed') & strcmp(indparam, 'free')
           trops(3) = 2;
       end
    else
       trops(3) = -1;
    end
    [logtheta Xu mu s2 LogL, elapsedTrain, elapsedPred] = runSnelsoncode(logtheta0, Xtrain, Ytrain, Xtest, Xuinit, Iterations, trops(3));
    FITC.elapsedTrain(it,i) = elapsedTrain;
    FITC.elapsedPred(it,i) = elapsedPred;
    
    
    if strcmp(typeKL, 'Full')
    % compute the Snelson-Ghahramani approximation posterior 
    [v, Ksta] = feval(covfunc{:}, logtheta, Xtrain(1,:), Xtrain(1,:)); 
    [v, Knu] = feval(covfunc{:}, logtheta, Xtrain, Xu);
    [v, Kuu] = feval(covfunc{:}, logtheta, Xu, Xu);
    sigma2n = exp(2*logtheta(end));
    LF = jitterChol(Kuu)';
    LF = LF\eye(size(LF,1));
    Q = LF*Knu';
    Q = Q'*Q; 
    Lambda = Ksta - diag(Q) + sigma2n;
    invL = diag(1./Lambda);
    sqLambda = Lambda.^0.5;
    invLK = repmat(1./sqLambda(:), [1 size(Knu,2)]).*Knu; 
    Sigma = Kuu + invLK'*invLK;
    Sigma = jitterChol(Sigma)'; 
    Sigma = Sigma\eye(size(Sigma,1));  
    [v, KKnu] = feval(covfunc{:}, logtheta, Xtest, Xu);
    [v, Ktest] = feval(covfunc{:}, logtheta, Xtest, Xtest);
    Sigma = Sigma*KKnu';
    KSK = Sigma'*Sigma;
    Q = LF*KKnu';
    Q = Q'*Q; 
    Ktest = Ktest - Q + KSK;
    KtestL = jitterChol(Ktest)'; 
    %FITC.KLpq(it,i) = KL(Full_mutest, Full_Ktest, SGmu, SGK);
    %FITC.KLqp(it,i) = KL(SGmu, SGK, Full_mutest, Full_Ktest); 
    FITC.KLpq(it,i) = KLchol(Full_mutest, Full_KtestL, mu, KtestL);
    FITC.KLqp(it,i) = KLchol(mu, KtestL, Full_mutest, Full_KtestL); 
    end
    
    FITC.mse(it,i)  = errorMeasureRegress(mu, s2, Ytest, 'mse');
    FITC.nlp(it,i)  = errorMeasureRegress(mu, s2, Ytest, 'nlp');
    FITC.smse(it,i) = errorMeasureRegress(mu, s2, Ytest, 'smse');
    FITC.snlp(it,i) = errorMeasureRegress(mu, s2, Ytest, 'snlp', Ytrain);
    FITC.logL(it,i) = LogL; 
    
    FITC.KLpqDiag(it,i) = KLdiag(Fullmu, Fulls2, mu, s2);
    FITC.KLqpDiag(it,i) = KLdiag(mu, s2, Fullmu, Fulls2);
    FITC.params(:,it,i) = logtheta';     
    %  
  end
  % save the results regularly 
  save([outdir 'SparseGP' dataSet '_Hyp' hyperparam '_Ind' indparam '.mat'], 'Repititions', 'M', 'DTC', 'VAR', 'SD', 'FITC');
  %
end
