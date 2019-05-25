function [y_mean, y_std, y_ci, y_local] = localgp(x_new, x, y, varargin)
% localgp	Local regression by fitting a GP to the K-nearest neighbors
% 
%   [y_mean, y_std, y_ci, y_local] = localgp(x_new, x, y, varargin)

%{
% Options
ip = inputParser;
ip.addParameter('K', 100)
ip.addParameter('GPOptions', {})
ip.addParameter('Searcher', [])
ip.parse(varargin{:});

% Extract variables
K          = ip.Results.K;
gp_options = ip.Results.GPOptions;
searcher   = ip.Results.Searcher;
%}

% Optional values
names    = { 'K' , 'Searcher' };
defaults = {  100,  []        };
[K, searcher, ~, gp_options] = internal.stats.parseArgs(names, defaults, varargin{:});

% Size
d_y     = size(y, 2);
n       = size(x, 1);
n_x_new = size(x_new, 1);

% Number of Neighbors
K = min( K, n );

% Initialize output
y_mean = NaN(n_x_new, d_y);
y_std  = NaN(n_x_new, d_y);
y_ci   = NaN(n_x_new, 2, d_y);

if d_y == 1
    % If Y is n-by-1
    if isempty(searcher)
        idx = knnsearch(x, x_new, 'K', K);
    else
        idx = knnsearch(searcher, x_new, 'K', K);
    end
    x_cell = num2cell(x, 2);
    x_nn = x_cell( idx );
    y_nn = y( idx );
    
    if ~isequal(size(x_nn), size(idx)) % n_x_new == 1
        x_nn = x_nn';
        y_nn = y_nn';
    end
    
    if n_x_new == 1
        % Training data
        x_local  = vertcat( x_nn{1,:} );
        y_local  = y_nn(1,:)';
        % Fit GP
        gp_local = fitrgp(x_local, y_local, gp_options{:});
        [y_mean, y_std, y_ci] = predict(gp_local, x_new);
        
    else
        for i = 1:n_x_new
            local_gp_options = gp_options;
            % Training data
            x_local  = vertcat( x_nn{i,:} );
            y_local  = y_nn(i,:)';
            % Fit GP
            gp_local = fitrgp(x_local, y_local, local_gp_options{:});
            [y_mean(i,:), y_std(i,:), y_ci(i,:)] = predict(gp_local, x_new(i,:));
        end
    end
    
else
    % If Y is n-by-d_y
    parfor i = 1:d_y
        [y_mean(:,i), y_std(:,i), y_ci(:,:,i), y_local(:,i)] = localgp(x_new, x, y(:,i), varargin{:});
    end
end


end