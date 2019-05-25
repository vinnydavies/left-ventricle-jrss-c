close all
clear
clc

names = {'EmulationOutput_ObjectiveLogLoss', ...
         'EmulationLoss_ObjectiveLogLoss', ...
         'EmulationLoss_ObjectiveLoss', ...
         'EmulationLogLoss_ObjectiveLogLoss'};
     
for i = 1:length(names)
    [~, err(i)] = process_estimates(names{i}, 1, false); close all;
    err_euclid(i,1) = median(err(i).euclid);
    err_mahal(i,1)  = median(err(i).mahal);
    err_euclid_std(i,1) = iqr(err(i).euclid);
    err_mahal_std(i,1)  = iqr(err(i).mahal);
end

% Make table

err_table = table(err_euclid, err_euclid_std, err_mahal, err_mahal_std, 'RowNames', names);

% Format table

fprintf( ' \t \t \t     Euclidean \t\t  Mahalanobis\n ' ) 
fprintf(' ----------------------------------------------- \n ')
tmp = err_table{'EmulationOutput_ObjectiveLogLoss', :};
fprintf( ' Output \t| %6.4f (%6.4f) |\t%6.4f (%6.4f)\n ', tmp(1), tmp(2), tmp(3), tmp(4) ) 
tmp = err_table{'EmulationLoss_ObjectiveLogLoss', :};
fprintf( ' Loss \t \t| %6.4f (%6.4f) |\t%6.4f (%6.4f)\n ', tmp(1), tmp(2), tmp(3), tmp(4) ) 
tmp = err_table{'EmulationLogLoss_ObjectiveLogLoss', :};
fprintf( ' Log Loss \t| %6.4f (%6.4f) |\t%6.4f (%6.4f)\n ', tmp(1), tmp(2), tmp(3), tmp(4) ) 