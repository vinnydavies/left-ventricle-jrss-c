function varargout = sfdesign(N, D, Type, lb, ub)
% sfdesign   Returns a space filling design X in [0, 1]^D
% 
% SYNTAX
%   X = sfdesign(N, D, Type)
% 
% N: Number of points
% D: Dimensionality
% Type: 'rand', 'sobol', 'halton', 'latin'

if nargin < 3 || isempty(Type)
    Type = 'sobol';
end
if nargin < 4
    lb = [];
end
if nargin < 5
    ub = [];
end

switch Type
    case 'rand'
        X = rand(N, D);
    case 'sobol'
        S = sobolset(D, 'Skip', 1e3, 'Leap', 1e2);
        S = scramble(S, 'MatousekAffineOwen');
        X = net(S, N);
    case 'halton'
        S = haltonset(D, 'Skip', 1e3, 'Leap', 1e2);
        S = scramble(S, 'RR2');
        X = net(S, N);
    case 'latin'
        X = lhsdesign(N,D);
    case 'linspace'
        assert( length(lb) == D, 'Bounds dimensionality not equal D' )
        n_grid = round(nthroot(N, D));
        x_grid = cell(1,D);
        x_mat  = cell(1,D);
        x_vec  = cell(1,D);
        for i = 1:D
            x_grid{i} = linspace(lb(i), ub(i), n_grid)';
        end
        [x_mat{1:D}] = ndgrid(x_grid{:});
        for i = 1:D
            x_vec{i} = x_mat{i}(:);
        end
        X = horzcat(x_vec{:});
end

if strcmp(Type, 'linspace')
    varargout = {X, x_mat, x_vec};
else
    if ~isempty(lb) && ~isempty(ub)
        X = rescale(X, lb, ub);
    end
    varargout = {X};
end

end