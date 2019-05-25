%% Outputs Emulation

% Clean workspace
close all
clear
clc


%% Load data

rng default
load('Simulations4D')

do_plot = true;


%% Variables

x_train = XTrain4D.Variables;
y_train = YTrain4D.Variables;

[N, D] = size(x_train);
K      = size(y_train, 2);


%% Open parallel pool

clust = parcluster;
clust.NumWorkers = K;
parpool(K);


%% Plot data

if do_plot
    figure(1)
    for i = 1:25
        clf
        histogram(y_train(:,i))
        
        title(sprintf('Strain %d', i - 1))
        if i == 1
            title('Volume')
        end
        
        drawnow
        pause
        
    end
end


%% Model

lb = 0.1 * ones(1,D);
ub = 5 * ones(1,D);

nn_searcher = createns(x_train);

gp_options  = {'K', 100, ...
    'Searcher', nn_searcher, ...
    'KernelFunction', 'ardsquaredexponential'};


%% Test predictive power

x_test = XTest4D.Variables;
y_test = YTest4D.Variables;

% Predict a test points
predictions = localgp(x_test, x_train, y_train, gp_options{:});


%% Plot all

if do_plot
    figure(1)
    for i = 1:25
        subplot(5,5,i)
        scatter(y_test(:,i), predictions(:,i), 5, 'filled')
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
end

%% One at a time

if do_plot
    figure(1)
    
    for i = 1:25
        clf
        scatter(y_test(:,i), predictions(:,i), 'filled')
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
end



%% Load data

[n_test, d] = size(x_test);
estimates = NaN(n_test, d);

% Covariance
cov_mat = cov(y_train);

% Objective
nn_searcher = createns(x_train);
gp_options = {'K', 10, ...
    'Searcher', nn_searcher, ...
    'KernelFunction', 'ardmatern52', ...
    'Sigma', 1e-3};

for i = 1:n_test
    
    data   = y_test(i,:);
    x_true = x_test(i,:);
    
    % Data = [LVVolumeMRI, strainMRITotal(:)'];
        
    % Objective
    obj_fcn = @(x_new) localgp_loss_mahal(x_new, data, cov_mat, x_train, y_train, ...
        gp_options{:});
    
    % Starting point
    x0 = mean([ub; lb]);
    
    % Optimize
    optim_method = 'ps';
    
    tic
    switch optim_method
        case 'gs'
            solver  = GlobalSearch('Display', 'iter');
            problem = createOptimProblem('fmincon', ...
                'objective', obj_fcn, ...
                'lb', lb, ...
                'ub', ub, ...
                'x0', x0);
            [x_best, f_best] = run(solver, problem);
        case 'ps'
            [x_best, f_best] = patternsearch(obj_fcn,x0,[],[],[],[],lb,ub);
    end
    toc
    
    estimates(i,:) = x_best;
    
end