
dirr = 'results/';
Dataset = 'SnelsonToy1'; 
printResults = 1;

% load data
Xtrain = load('datasets/train_inputs');
Ytrain = load('datasets/train_outputs');
Xtest = load('datasets/test_inputs');
my = mean(Ytrain); 
Ytrain = Ytrain - my;
[n D] = size(Xtrain);

% type of likelihood 
options.Likelihood = 'Gaussian';

%number of inducing variables.
options.m = 10; 

% type of inducing variables
options.indType = 'pseudoIns'; % 'pseudoIns' or 'weights'
options.objectFunc = 'dtc';

model = varsgpCreate('seard', Xtrain, Ytrain, options);
% Fix seeds
randn('seed', 2e5);
rand('seed', 2e5);
Xuinit = mean(Xtrain(:)) + 0.5*randn(model.m,1); 
model.Xu = Xuinit;

trops(1) = 1000; % number of iterations
trops(2) = 1; % type of optimizations
trops(3) = 1; 

% initialization of the model hyperparameters 
logtheta0(1:D,1) = log((max(Xtrain)-min(Xtrain))'/2);
logtheta0(D+1,1) = 0.5*log(var(Ytrain,1)); 
logtheta0(D+2,1) = 0.5*log(var(Ytrain,1)/4);  
model.GP.logtheta = logtheta0(1:end-1);
model.Likelihood.logtheta = logtheta0(end);


% train the model by optimizing over the kernel hyperparameters  
% the inducing variables parameters (pseudo inputs or weights)
[model margLogL] = varsgpTrain(model, trops);

% prediction in test data
[mustar, varstar, covstar] = varsgpPredict(model, Xtest);
% add noise variance sigma2 to get a prediction for the ys
varstar = varstar + exp(2*model.Likelihood.logtheta);

% display prediction in test data
% run firstly the full GP to know the ground truth 
covfunc = {'covSum', {'covSEard','covNoise'}};
[logthetaFullGP, fullF] = minimize(logtheta0, 'gpr', -200, covfunc, Xtrain, Ytrain);
[muFull s2Full] = gpr(logthetaFullGP, covfunc, Xtrain, Ytrain, Xtest);
figure;
hold on;
fillColor = [0.8 0.8 0.8];
fill([Xtest; Xtest(end:-1:1)], [muFull; muFull(end:-1:1)]...
               + 2*[sqrt(s2Full); -sqrt(s2Full(end:-1:1))], fillColor,'EdgeColor',fillColor);
plot(Xtest, muFull,'b','lineWidth',3);

     
plot(Xtrain, Ytrain,'.k','markersize',12); % data points in magenta

plot(Xtest, mustar, 'r-.', 'LineWidth', 2) % mean predictions in blue
plot(Xtest, mustar + 2*sqrt(varstar),'-r','LineWidth',2) % plus/minus 2 std deviation
plot(Xtest, mustar - 2*sqrt(varstar),'-r','LineWidth',2) % plus/minus 2 std deviation

plot(model.Xu,-2.75*ones(size(model.Xu)),'k+','markersize',20,'LineWidth',1);
plot(Xuinit,2.2*ones(size(Xuinit)),'k+','markersize',20,'LineWidth',1);

axis([min(Xtest) max(Xtest) -3 3]);
set(gca, 'fontsize', 18);
%set(gca, 'YTick', []);
%set(gca, 'XTick', []);
box on;

if printResults
    print('-depsc',[dirr Dataset options.objectFunc]);
    print('-dpng', [dirr Dataset options.objectFunc]);
end
