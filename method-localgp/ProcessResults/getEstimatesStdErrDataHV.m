close all
clear
clc

estimFcn = @hessToCovMat;

logLoss = load('EmulationOutput_ObjectiveLogLoss_MethodGS_DataHV.mat');
loss    = load('EmulationOutput_ObjectiveLoss_MethodGS_DataHV.mat');

xll = logLoss.res.x_best_euclid(1,:);
xl  = loss.res.x_best_euclid(1,:);
xtrue = ones(1,4);
norm(xl - xtrue) < norm(xll - xtrue)


for ID = [1] % the other IDs have been removed
        
    H = logLoss.res.hess_euclid{ID};
    
    if false % all(isreal(estimFcn(H)))
        disp('logLoss')
        x = logLoss.res.x_best_euclid(ID,:);
        H = logLoss.res.hess_euclid{ID};
        cov_mat = estimFcn(H);
    else
        disp('loss')
        x = loss.res.x_best_euclid(ID,:);
        H = loss.res.hess_euclid{ID};
        cov_mat = estimFcn(H);
    end
        
    save(sprintf('hv_data_estimate_patient_%d.txt',    ID), 'x',       '-ascii')
    save(sprintf('hv_data_hess_matrix_patient_%d.txt', ID), 'H',       '-ascii')
    save(sprintf('hv_data_cov_matrix_patient_%d.txt',  ID), 'cov_mat', '-ascii')
    
end


function covMat = hessToCovMat(H)

covMat = inv(H);

end