% Tidy up results

close all
clear
clc

name = 'Hessian_EmulationOutput_ObjectiveLogLoss_MethodGS_DataTest_Row1';
n    = 100;

saveName = 'Hessian_EmulationOutput_ObjectiveLogLoss_MethodGS_DataTest';


%% Internal

tmp = load(name);
fields = fieldnames(tmp);
nFields = length(fields);

nameWithoutRowNumber = name(1:end-1);

for i = 1:n
    
    iFileName = sprintf('%s%d', nameWithoutRowNumber, i);
    iResults = load(iFileName);
    
    res.x_best_euclid(i,:) = iResults.x_best_euclid;
    res.f_best_euclid(i,:) = iResults.f_best_euclid;
    res.grad_euclid{i}     = iResults.grad_euclid;
    res.hess_euclid{i}     = iResults.hess_euclid;
    res.hess_euclid_bis{i} = iResults.hess_euclid_bis;
    
    res.x_best_mahal(i,:) = iResults.x_best_mahal;
    res.f_best_mahal(i,:) = iResults.f_best_mahal;
    res.grad_mahal{i}     = iResults.grad_mahal;
    res.hess_mahal{i}     = iResults.hess_mahal;
    res.hess_mahal_bis{i} = iResults.hess_mahal_bis;
end

save(saveName, 'res')

