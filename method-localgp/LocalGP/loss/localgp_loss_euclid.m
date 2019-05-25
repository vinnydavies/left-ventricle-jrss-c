function loss = localgp_loss_euclid(x_new, data, x, y, varargin)
% Euclidean loss
% 
%   loss = localgp_loss_euclid(x_new, data, x, y, ...)

assert( size(x_new, 1) == 1, 'Only for one test input' )

% Predict
y_hat = localgp(x_new, x, y, varargin{:});

% Calculate loss
loss = norm( y_hat - data ).^2;

end