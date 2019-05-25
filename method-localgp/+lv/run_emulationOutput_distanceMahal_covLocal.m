function run_emulationOutput_distanceMahal_covLocal( which_objective, which_data, id )


%% Prepare workspace

% Random seed
rng default

% Load data
load( fullfile('Simulations', 'Design4D', 'Simulations4D') );


%% Variables

% Training data
x_train = XTrain4D.Variables;
y_train = YTrain4D.Variables;

d = size(x_train, 2);

% Inferential data
switch which_data
    case 'test'
        % Inferential data
        y_test = YTest4D.Variables;
        data   = y_test(id,:);
    case 'hv'
        % Use hv data
        tmp  = load('DataHV');
        data = tmp.data(id,:);
end

% Standardize simulations
[y_train, y_train_mean, y_train_std] = zscore(y_train);

% Shift data
data = (data - y_train_mean) ./ y_train_std;


%% Open parallel pool

clust = parcluster;
clust.NumWorkers = 24;
parpool(24);


%% Optimize

% Bounds
lb = 0.1 * ones(1,d);
ub =   5 * ones(1,d);

% Options for localgp
nn_searcher = createns(x_train);
gp_options  = {'K', 100, ...
    'Searcher', nn_searcher, ...
    'KernelFunction', 'ardsquaredexponential', ...
    'Sigma', 1e-2};

% Objective
cov_type = 'local';

switch which_objective
    case 'loss'
        loss_mahal_fcn  = @(x_new) ...
            localgp_loss_mahal(x_new, data, cov_type, x_train, y_train, gp_options{:});
    case 'log_loss'
        loss_mahal_fcn  = @(x_new) ...
            log( localgp_loss_mahal(x_new, data, cov_type, x_train, y_train, gp_options{:}) + 1 );
end

% Starting point
x0 = mean([ub; lb]);

% Global search
gs = GlobalSearch('NumTrialPoints', 2000, 'NumStageOnePoints', 400);
prob_mahal = createOptimProblem('fmincon', ...
    'objective', loss_mahal_fcn, ...
    'x0', x0, ...
    'lb', lb, ...
    'ub', ub);
[x_best_mahal,  f_best_mahal]  = run(gs, prob_mahal);

% Gradient
hess_mahal  = hessian(loss_mahal_fcn,  x_best_mahal);


%% Save

% Process filename parts
data_string_save      = capitalize( which_data );
objective_string_save = capitalize( split(which_objective, '_') );
objective_string_save = horzcat( objective_string_save{:} );

% Filename
save_name = sprintf('EmulationOutput_CovLocal_Objective%s_DistanceMahal_MethodGS_Data%s_Row%d', ...
    objective_string_save, data_string_save, id);

% Save
save( fullfile('Results', 'EmulationOutput', 'SigmaInit1e-2', save_name), ...
    'x_best_mahal',  'f_best_mahal',  'hess_mahal' )


end

