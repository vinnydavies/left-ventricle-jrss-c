function loss = localgp_loss_mahal(x_new, data, cov_type, x, y, varargin)
% Mahalanobis loss
% 
%   loss = localgp_loss_mahal(x_new, data, cov_type, x, y, ...)

assert( size(x_new, 1) == 1, 'Only for one test input' )

% Predict
[y_hat, ~, ~, y_local] = localgp(x_new, x, y, varargin{:});

% Cov matrix
switch cov_type
    case 'local'
        cov_mat = cov(y_local);
    case 'full'
        cov_mat = cov(y);
    otherwise
        error('cov_type can be local or full')
end

% Loss
loss = (y_hat - data) * (cov_mat \ (y_hat - data)');

end