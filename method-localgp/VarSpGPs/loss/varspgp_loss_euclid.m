function loss = varspgp_loss_euclid(mdl, x_new, data, varargin)
% Euclidean loss
% 
%   loss = varspgp_loss_euclid(mdl, x_new, data, varargin)

assert( size(x_new, 1) == 1, 'Only for one test input' )

% Predict
y_hat = varspgp_predict(mdl, x_new, varargin{:});

% Calculate loss
loss = norm( y_hat - data ).^2;

end