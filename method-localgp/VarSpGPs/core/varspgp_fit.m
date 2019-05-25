function [model, margLogL] = varspgp_fit(x_train, y_train, varargin)

% Defaults
names = {'n_xu', 'init_xu', 'n_iter', 'parallel'};
deflt = {    [],        [],       [],      false};
[n_xu, init_xu, n_iter, parallel] = internal.stats.parseArgs(names, deflt, varargin{:});

% Fit model
n_species = size(y_train, 2);
model     = cell(1,n_species);
margLogL  = cell(1,n_species);

if ~parallel
    for i = 1:n_species
        [model{i}, margLogL{i}] = i_varspgp_fit(x_train, y_train(:,i), n_xu, init_xu, n_iter);
    end
else
    parfor i = 1:n_species
        [model{i}, margLogL{i}] = i_varspgp_fit(x_train, y_train(:,i), n_xu, init_xu, n_iter);
    end
end

end


function [model, margLogL] = i_varspgp_fit(x_train, y_train, n_xu, init_xu, n_iter)

% Number of training inputs
[n_train, D] = size(x_train);

% type of likelihood 
options.Likelihood = 'Gaussian';

% number of inducing variables.
if nargin < 3 || isempty(n_xu)
    n_xu = min(100, n_train);
end
options.m = n_xu; 

% type of inducing variables
options.indType = 'pseudoIns'; % 'pseudoIns' or 'weights'
options.objectFunc = 'var';

% create variational sparse gp
model = varsgpCreate('seard', x_train, y_train, options);

if nargin < 4 || isempty(init_xu)
    init_xu = new_rescale(sfdesign(options.m, D), min(x_train), max(x_train));
end
model.Xu = init_xu;

if nargin < 5 || isempty(n_iter)
    n_iter = 1000;
end
trops(1) = n_iter; % number of iterations
trops(2) = 1; % type of optimizations
trops(3) = 1; 

% initialization of the model hyperparameters 
logtheta0(1:D,1) = log((max(x_train)-min(x_train))'/2);
logtheta0(D+1,1) = 0.5*log(var(y_train,1)); 
logtheta0(D+2,1) = 0.5*log(var(y_train,1)/4);  
model.GP.logtheta = logtheta0(1:end-1);
model.Likelihood.logtheta = logtheta0(end);

% train the model by optimizing over the kernel hyperparameters  
% the inducing variables parameters (pseudo inputs or weights)
[model, margLogL] = varsgpTrain(model, trops);

end
