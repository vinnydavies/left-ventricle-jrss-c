function [mustar, varstar] = varspgp_predict(mdl, x_test, varargin)

% Defaults
parallel = internal.stats.parseArgs({'parallel'}, {false}, varargin{:});

% Predict from model
n_species = length(mdl);
n_test    = size(x_test, 1);
mustar    = NaN(n_test, n_species);
varstar   = NaN(n_test, n_species);

if ~parallel
    for i = 1:n_species
        [mustar(:,i), varstar(:,i)] = varsgpPredict(mdl{i}, x_test);
    end
else
    parfor i = 1:n_species
        [mustar(:,i), varstar(:,i)] = varsgpPredict(mdl{i}, x_test);
    end
end

end