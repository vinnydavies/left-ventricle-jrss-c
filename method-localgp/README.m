% README for Local GPs


% Make sure that the current directory is 'method-localgp'

% Add folders to path
startup;

% 'test': data from test set (100 data points)
% 'hv': data from healthy volunteer (1 data point)
data_label = 'test';

% Create the right Results folder, see at the end of lv.run_emulation_*
% Example: method-localgp/Results/*

% For a given dataset data_label, identify the data point.
% This allows for estimation in parallel independent jobs on a computer cluster, 
% by setting the data_id to the PBS array ID.
% For data_label = 'test', data_id can be 1, ..., 100. For data_label = 'hv', data_id = 1.
data_id = 1;

% Run emulation
lv.run_emulation_log_loss(data_label, data_id);

