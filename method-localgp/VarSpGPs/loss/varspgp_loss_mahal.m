function loss = varspgp_loss_mahal(mdl, x_new, data, cov_mat, varargin)
% Mahalanobis loss
% 
%   loss = varspgp_loss_mahal(mdl, x_new, data, cov_mat, ...)

assert( size(x_new, 1) == 1, 'Only for one test input' )

% Predict
y_hat = varspgp_predict(mdl, x_new, varargin{:});

% Loss
loss = (y_hat - data) * (cov_mat \ (y_hat - data)');

end