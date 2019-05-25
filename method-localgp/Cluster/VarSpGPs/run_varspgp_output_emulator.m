function run_varspgp_output_emulator( id, which_data )


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
        tmp  = load('hv_data');
        data = tmp.data(id,:);
end

% Standardize
[y_train, y_train_mean, y_train_std] = zscore(y_train);
data = (data - y_train_mean) ./ y_train_std;


%% Open parallel pool

% clust = parcluster;
% clust.NumWorkers = 24;
% parpool(24);


%% Optimize

% Bounds
lb = 0.1 * ones(1,d);
ub =   5 * ones(1,d);

% Options for localgp
gp_mdl = varspgp_fit(x_train, y_train, 'n_xu', 500, 'n_iter', 1000);

% Objective
cov_mat = cov(y_train);
loss_mahal_fcn  = @(x_new) localgp_loss_mahal (x_new, data, cov_mat, x_train, y_train, gp_options{:});
loss_euclid_fcn = @(x_new) localgp_loss_euclid(x_new, data, x_train, y_train, gp_options{:});

% Starting point
x0 = mean([ub; lb]);

% Global search
gs = GlobalSearch('NumTrialPoints', 25000, 'NumStageOnePoints', 1000);
prob_mahal = createOptimProblem('fmincon', 'objective', loss_mahal_fcn, ...
    'x0', x0, 'lb', lb, 'ub', ub);
prob_euclid = createOptimProblem('fmincon', 'objective', loss_euclid_fcn, ...
    'x0', x0, 'lb', lb, 'ub', ub);
[x_best_mahal,  f_best_mahal]  = run(gs, prob_mahal);
[x_best_euclid, f_best_euclid] = run(gs, prob_euclid);

% Gradient
grad_mahal  = gradest(loss_mahal_fcn,  x_best_mahal);
grad_euclid = gradest(loss_euclid_fcn, x_best_euclid);


%% Save

save_name = sprintf('estimates_output_emulation_%s_%d', which_data, id);
save( fullfile('Results', 'OutputEmulation', 'SigmaInitDefault', save_name), ...
    'x_best_mahal',  'f_best_mahal',  'grad_mahal', ...
    'x_best_euclid', 'f_best_euclid', 'grad_euclid' )


end

