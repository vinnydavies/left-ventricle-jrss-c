function [test, err] = process_estimates(varargin)
% Analysis of LV estimates
% 
%   [test, err] = process_estimates(varargin)


%% Defaults

names = {'file', 'distance', 'error_power', 'show_plots'};
dflts = {    '',         '',             1,         true};

[file_name, distance, error_power, show_plots] = ...
    internal.stats.parseArgs(names, dflts, varargin{:});

assert(~isempty(file_name), 'File path required');
assert(~isempty(distance), 'Distance required');
assert( (error_power >= 1 && error_power <= 2), ...
    'error_power must be 1 or 2' )


%% Variables

read_euclid = ismember(distance, {'euclid', 'all'});
read_mahal  = ismember(distance, {'mahal' , 'all'});

switch error_power
    case 1
        error_string = 'Absolute';
    case 2
        error_string = 'Squared';
end

title_string = lv.make_title(file_name);


%% Load data

load('Simulations4D', 'XTest4D')
x_test_true = XTest4D.Variables;


%% Estimation results

test_struct = load(file_name);
test = test_struct.res;

if read_euclid
    x_best_euclid = test.x_best_euclid;
end
if read_mahal
    x_best_mahal = test.x_best_mahal;
end


%% Loss boxplots

% Euclidean loss
if show_plots && read_euclid
    figure
    boxplot(x_test_true - x_best_euclid)
    
    xlabel('Theta')
    ylabel('True - Estimated')
    title(sprintf('%s, Distance Euclidean', title_string))
    set(gca, 'FontSize', 15)
    
    yl = ylim;    
end

% Mahalanobis loss
if show_plots && read_mahal
    figure
    boxplot(x_test_true - x_best_mahal)
    
    xlabel('Theta')
    ylabel('True - Estimated')
    title(sprintf('%s, Distance Mahalanobis', title_string))
    
    set(gca, 'FontSize', 15)
    % ylim(yl)
end


%% Absolute errors Euclidean

if read_euclid
    % Calculate error
    euclid_err      = abs( x_test_true - test.x_best_euclid ) .^ error_power;
    euclid_mean_err = mean(euclid_err, 2);
    
    % Plot
    if show_plots
        figure
        set(gcf, 'Position', [300 300 1300 500])
        
        subplot(1,2,1)
        boxplot( euclid_err )
        xlabel('Theta')
        ylabel( sprintf('%s Error Distribution', error_string) )
        set(gca, 'FontSize', 15)
        
        subplot(1,2,2)
        boxplot( euclid_mean_err )
        xlabel('Theta')
        ylabel( sprintf('Mean %s Error Distribution', error_string) )
        set(gca, 'FontSize', 15)
        
        suptitle(sprintf('%s, Distance Euclidean', title_string))
    end
end


%% Absolute errors Mahalanobis

if read_mahal
    % Calculate error
    mahal_err      = abs( x_test_true - test.x_best_mahal ) .^ error_power;
    mahal_mean_err = mean(mahal_err, 2);
    
    % Plot
    if show_plots
        figure
        set(gcf, 'Position', [300 300 1300 500])
        
        subplot(1,2,1)
        boxplot( mahal_err )
        xlabel('Theta')
        ylabel( sprintf('%s Error Distribution', error_string) )
        set(gca, 'FontSize', 15)
        
        subplot(1,2,2)
        boxplot( mahal_mean_err )
        xlabel('Theta')
        ylabel( sprintf('Mean %s Error Distribution', error_string) )
        set(gca, 'FontSize', 15)
        
        suptitle( sprintf('%s, Distance Mahalanobis', title_string) )
    end
end


%% Error measures

if read_euclid
    err.euclid = euclid_mean_err;
end
if read_mahal
    err.mahal  = mahal_mean_err;
end


end
