% README for Local GPs


% Make sure that the current directory is 'method-localgp'

% Add folders to path
startup;

% 'test': data from test set
% 'hv': data from healthy volunteer
data_label = 'test';

% Create the right Results folder, see at the end of lv.run_emulation_*
% Example: method-localgp/Results/*

% For independent parallel jobs data_id should be the PBS array ID.
% See in the Cluster folder
data_id = 1;

% Run emulation
lv.run_emulation_log_loss(data_label, data_id);

