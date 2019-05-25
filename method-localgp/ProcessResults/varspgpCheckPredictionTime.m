%% Prepare workspace

close all
clear
clc

% Random seed
rng default

% Load data
load( fullfile('Simulations', 'Design4D', 'Simulations4D') );

which_data = 'test';
id = 1;

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
        x_test = XTest4D.Variables;
    case 'hv'
        % Use hv data
        tmp  = load('hv_data');
        data = tmp.data(id,:);
end

% Standardize
[y_train, y_train_mean, y_train_std] = zscore(y_train);
data = (data - y_train_mean) ./ y_train_std;

y_test = (y_test - y_train_mean) ./ y_train_std;

%% Open parallel pool

clust = parcluster;
clust.NumWorkers = 24;
parpool(24);


%% Fit model

% Options for localgp
gp_mdl = varspgp_fit(x_train, y_train, 'n_xu', 500, 'n_iter', 1000, 'parallel', true);
y_hat  = varspgp_predict(gp_mdl, x_test);


%% Plot

figure(1)
set(gcf, 'Position', [100 50 1200 950])
for i = 1:25
    subplot(5,5,i)
    scatter(y_test(:,i), y_hat(:,i), 5, 'filled')
    axis tight
    
    hold on
    xl = xlim;
    yl = ylim;
    plot(xl, yl, 'r-')
    hold off
    
    title(sprintf('Strain %d', i - 1))
    if i == 1
        title('Volume')
    end
    ylabel('Predicted')
    xlabel('True')
end

%% Plot one at a time

figure(1)
set(gcf, 'Position', [100 50 1200 950])
for i = 1:25
    clf;
    scatter(y_test(:,i), y_hat(:,i), 'filled')
    axis tight
    
    hold on
    xl = xlim;
    yl = ylim;
    plot(xl, yl, 'r-')
    hold off
    
    title(sprintf('Strain %d', i - 1))
    if i == 1
        title('Volume')
    end
    ylabel('Predicted')
    xlabel('True')
    drawnow
    pause
end


%%

tic
x = rand(1,4);
cellfun(@(idml) varsgpPredict(idml, x), gp_mdl);
toc



surrogate1 = @(x) cellfun(@(idml) varsgpPredict(idml, x), gp_mdl);
tic; surrogate1(rand(1,4)); toc
tic; surrogate1(rand(1,4)); toc



surrogate2 = @(x) [ ...
    varsgpPredict(gp_mdl{1}, x), ...
    varsgpPredict(gp_mdl{2}, x), ...
    varsgpPredict(gp_mdl{3}, x), ...
    varsgpPredict(gp_mdl{4}, x), ...
    varsgpPredict(gp_mdl{5}, x), ...
    varsgpPredict(gp_mdl{6}, x), ...
    varsgpPredict(gp_mdl{7}, x), ...
    varsgpPredict(gp_mdl{8}, x), ...
    varsgpPredict(gp_mdl{9}, x), ...
    varsgpPredict(gp_mdl{10}, x), ...
    varsgpPredict(gp_mdl{11}, x), ...
    varsgpPredict(gp_mdl{12}, x), ...
    varsgpPredict(gp_mdl{13}, x), ...
    varsgpPredict(gp_mdl{14}, x), ...
    varsgpPredict(gp_mdl{15}, x), ...
    varsgpPredict(gp_mdl{16}, x), ...
    varsgpPredict(gp_mdl{17}, x), ...
    varsgpPredict(gp_mdl{18}, x), ...
    varsgpPredict(gp_mdl{19}, x), ...
    varsgpPredict(gp_mdl{20}, x), ...
    varsgpPredict(gp_mdl{21}, x), ...
    varsgpPredict(gp_mdl{22}, x), ...
    varsgpPredict(gp_mdl{23}, x), ...
    varsgpPredict(gp_mdl{24}, x), ...
    varsgpPredict(gp_mdl{25}, x) ];
tic; surrogate2(rand(1,4)); toc
tic; surrogate2(rand(1,4)); toc



%%

% Bounds
lb = 0.1 * ones(1,d);
ub =   5 * ones(1,d);

% Starting point
x0 = mean([ub; lb]);

% Objective
cov_mat = cov(y_train);
loss_mahal_fcn  = @(x_new) varspgp_loss_mahal (gp_mdl, x_new, data, cov_mat, 'parallel', true);
loss_euclid_fcn = @(x_new) varspgp_loss_euclid(gp_mdl, x_new, data);

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


