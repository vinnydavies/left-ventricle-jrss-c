%% Analysis of results


%% Prepare workspace

close all
clear
clc

%% Load data

load('Simulations4D')
x_test_true = XTest4D.Variables;

%% Estimation results

data = stack_estimates_lv('estimates_hv_data_log_', 9, 4);
test = stack_estimates_lv('estimates_test_data_log_', 100, 4);

%% Boxplot

print_plot = false;

% Euclidean loss
figure
boxplot(x_test_true - test.x_best_euclid)
title('Euclidean loss')
ylabel('True - Estimated')
xlabel('Theta')
set(gca, 'FontSize', 15)
yl = ylim;
if print_plot
    print( fullfile('Documents', 'Figures', 'OutputEmulationEuclidLoss'), '-depsc' )
end

% Mahalanobis loss
figure
boxplot(x_test_true - test.x_best_mahal)
title('Mahalanobis loss')
ylabel('True - Estimated')
xlabel('Theta')
set(gca, 'FontSize', 15)
% ylim(yl)
if print_plot
    print( fullfile('Documents', 'Figures', 'OutputEmulationMahalLoss'), '-depsc' )
end

%% Absolute errors

abs_err      = abs( x_test_true - test.x_best_mahal );
mean_abs_err = mean( abs( x_test_true - test.x_best_mahal ), 2);

figure
boxplot( abs_err )
title('Absolute Error in Output Emulation - Mahalanobis loss')
xlabel('Theta')
ylabel('Absolute Error Distribution')


figure
boxplot( mean_abs_err )
title('Mean Absolute Error in Output Emulation - Mahalanobis loss')
xlabel('Theta')
ylabel('Average Absolute Error Distribution')

