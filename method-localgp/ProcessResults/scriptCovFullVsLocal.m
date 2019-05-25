close all
clear
clc

distance = 'mahal';

[~, errCovFull] = lv.process_estimates('File', 'EmulationOutput_ObjectiveLogLoss_MethodGS_DataTest', ...
    'Distance', distance);

[~, errCovLocal] = lv.process_estimates('File', 'EmulationOutput_CovLocal_ObjectiveLogLoss_DistanceMahal_MethodGS_DataTest', ...
    'Distance', distance);

errCovFull  = errCovFull.(distance);
errCovLocal = errCovLocal.(distance);

table( [median(errCovFull), median(errCovLocal)]', [iqr(errCovFull), iqr(errCovLocal)]', ...
    'VariableNames', {'Median', 'IQR'}, 'RowNames', {'Full', 'Local'})