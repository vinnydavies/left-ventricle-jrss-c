
close all
clear
clc

fileName1 = 'EmulationOutput_CovLocal_ObjectiveLogLoss_DistanceMahal_MethodGS_DataTest';
fileName2 = 'EmulationOutput_ObjectiveLogLoss_MethodGS_DataTest';

lv.process_estimates('file', fileName1, 'distance', 'mahal')
lv.process_estimates('file', fileName2, 'distance', 'mahal')
