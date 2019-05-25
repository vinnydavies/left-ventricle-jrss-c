%% Outputs Emulation

% Clean workspace
close all
clear
clc
rng default

% Load data
load('Simulations4D')

% Settings
which_data = 'test';  % hv or test
id         = 7;

log_data   = true;
den_m      = @(d) 1;
den_e      = @(d) 1;


%% Simulations

% Training data
x_train = XTrain4D.Variables;
y_train = YTrain4D.Variables;

[n_train, d] = size(x_train);

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

% Standardize simulations
% [y_train, y_train_mean, y_train_std] = zscore(y_train);

% Shift data
% data = (data - y_train_mean) ./ y_train_std;


%% Loss

% Initialize loss
loss_euclid_train = NaN(n_train, 1);
loss_mahal_train  = NaN(n_train, 1);
cov_mat = cov(y_train);

% Calculate losses
for i = 1:n_train
    % Euclidean
    loss_euclid_train(i,1) = norm( y_train(i,:) - data ).^2;
    % Mahalanobis
    loss_mahal_train(i,1)  = ( y_train(i,:) - data ) * (cov_mat \ ( y_train(i,:) - data )');
end

% Normalize loss
loss_euclid_train = loss_euclid_train / den_e(data);
loss_mahal_train  = loss_mahal_train  / den_m(data);

[loss_euclid_train, letm, lets] = zscore(loss_euclid_train);
[loss_mahal_train, lmtm, lmts]  = zscore(loss_mahal_train);

if log_data
    % Log training data
    loss_euclid_train = log(loss_euclid_train + 1);
    loss_mahal_train  = log(loss_mahal_train + 1);
end



%% Model

% Options for localgp
nn_searcher = createns(x_train);
gp_options  = {'K', 100, ...
    'Searcher', nn_searcher, ...
    'KernelFunction', 'ardsquaredexponential', 'Sigma', 1e-2};

% Euclid loss
loss_euclid_fcn = @(x_new) localgp(x_new, x_train, loss_euclid_train, gp_options{:});

% Mahal loss
loss_mahal_fcn  = @(x_new) localgp(x_new, x_train, loss_mahal_train, gp_options{:});


%% Test data

x_test = XTest4D.Variables;
y_test = YTest4D.Variables;
x_test(id,:) = [];
y_test(id,:) = [];
n_test = size(x_test, 1);

% Shift data
% y_test = (y_test - y_train_mean) ./ y_train_std;

% Initialize loss
loss_euclid_test = NaN(n_test, 1);
loss_mahal_test  = NaN(n_test, 1);

% Test losses
for i = 1:n_test
    % Euclidean
    loss_euclid_test(i,1) = norm( y_test(i,:) - data ).^2;
    % Mahalanobis
    loss_mahal_test(i,1)  = ( y_test(i,:) - data ) * (cov_mat \ ( y_test(i,:) - data )');
end

% Normalize loss
loss_euclid_test = loss_euclid_test / den_e(data);
loss_mahal_test  = loss_mahal_test  / den_m(data);

loss_euclid_test = (loss_euclid_test - letm) ./ lets;
loss_mahal_test  = (loss_mahal_test - lmtm) ./ lmts;

if log_data
    loss_euclid_test = log(loss_euclid_test + 1);
    loss_mahal_test  = log(loss_mahal_test + 1);
end



%% Predict at test points

pred_euclid_test = loss_euclid_fcn(x_test);
pred_mahal_test  = loss_mahal_fcn(x_test);

norm( pred_euclid_test - loss_euclid_test )^2
norm( pred_mahal_test - loss_mahal_test )^2


%% Plot all

figure

subplot(1,2,1)
scatter(loss_euclid_test, pred_euclid_test, 10, 'filled')
% set(gca, 'yscale', 'log', 'xscale', 'log')
axis tight
hold on
xl = xlim;
yl = ylim;
plot(xl, yl, 'r-')
hold off
title('Euclidean loss')
xlabel('True')
ylabel('Predicted')

subplot(1,2,2)
scatter(loss_mahal_test, pred_mahal_test, 10, 'filled')
% set(gca, 'yscale', 'log', 'xscale', 'log')
axis tight
hold on
xl = xlim;
yl = ylim;
plot(xl, yl, 'r-')
hold off
title('Mahalanobis loss')
xlabel('True')
ylabel('Predicted')
