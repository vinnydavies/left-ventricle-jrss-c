function run_hessian_output( which_objective, which_data, id )


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

% Process filename parts
data_string_save      = capitalize( which_data );
objective_string_save = capitalize( split(which_objective, '_') );
objective_string_save = horzcat( objective_string_save{:} );

% Starting points
x0_name = sprintf('EmulationOutput_Objective%s_MethodGS_Data%s', ...
    objective_string_save, data_string_save);
x0_struct = load(x0_name);

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
cov_type = 'full';

switch which_objective
    case 'loss'
        loss_mahal_fcn  = @(x_new) ...
            localgp_loss_mahal(x_new, data, cov_type, x_train, y_train, gp_options{:});
        loss_euclid_fcn = @(x_new) ...
            localgp_loss_euclid(x_new, data, x_train, y_train, gp_options{:});
    case 'log_loss'
        loss_mahal_fcn  = @(x_new) ...
            log( localgp_loss_mahal(x_new, data, cov_type, x_train, y_train, gp_options{:}) + 1 );
        loss_euclid_fcn = @(x_new) ...
            log( localgp_loss_euclid(x_new, data, x_train, y_train, gp_options{:}) + 1 );
end

% Starting point
x0_euclid = x0_struct.res.x_best_euclid(id,:);
x0_mahal  = x0_struct.res.x_best_mahal(id,:);

% Minimize Euclid
[x_best_euclid,f_best_euclid,exitflag,output,lambda,grad_euclid,hess_euclid] = ...
    fmincon(loss_euclid_fcn, x0_euclid, [],[],[],[],lb,ub);

% Minimize Mahal
[x_best_mahal,f_best_mahal,exitflag,output,lambda,grad_mahal,hess_mahal] = ...
    fmincon(loss_mahal_fcn, x0_mahal, [],[],[],[],lb,ub);

% Gradient
hess_euclid_bis = hessian(loss_euclid_fcn, x_best_euclid);
hess_mahal_bis  = hessian(loss_mahal_fcn,  x_best_mahal);


%% Save

% Filename
save_name = sprintf('Hessian_EmulationOutput_Objective%s_MethodGS_Data%s_Row%d', ...
    objective_string_save, data_string_save, id);

% Save
save( fullfile('Results', 'EmulationOutput', 'SigmaInit1e-2', save_name), ...
    'x_best_mahal' , 'f_best_mahal' , 'grad_mahal' , 'hess_mahal' , 'hess_mahal_bis', ...
    'x_best_euclid', 'f_best_euclid', 'grad_euclid', 'hess_euclid', 'hess_euclid_bis' )


end

