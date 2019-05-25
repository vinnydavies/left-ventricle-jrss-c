addpath ../gpml-matlab/gpml/
addpath ../edsnelson/SPGP_dist/
addpath misc/


% run the Snelson toy  data;
demToyPseudoDtc1; demToyPseudoFitc1;  demToyPseudoVar1; 
demToyPseudoDtc2; demToyPseudoFitc2;  demToyPseudoVar2; 

% run in the Siwss rain fall data
%demSwissRainFallPseudoDtc1;  demSwissRainFallPseudoVar1; demSwissRainFallPseudoFitc1;

% repititions of the experiments (for only the sparse methods)
Repititions = 10;
% run on all the datasets
for Dataset =  {'Pendulum', 'Boston', 'Pumadyn32nm', 'Pol', 'Elevators','Kin40k', 'Abalone',}
     % % fixed hyperparameters and fixed pseudo inputs  
     % runDataset(Dataset, 'fixed', 'fixed', Repititions); 
     % % free hyperparameters and fixed pseudo inputs
     % runDataset(Dataset, 'free', 'fixed', Repititions); 
     % % fixed hyperparameters and free pseudo inputs
     % runDataset(Dataset, 'fixed', 'free', Repititions); 
     % free hyperparameters and free pseudo inputs
     runDataset(Dataset, 'free', 'free', Repititions);
end

% create the plots 
Dataset =  {'Pendulum', 'Boston', 'Pumadyn32nm', 'Pol', 'Elevators', 'Kin40k', 'Abalone'};
for j=1:7, createPlots(Dataset{j}, 1); end;


