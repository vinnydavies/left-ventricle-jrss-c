function experiment = stack_estimates(names, n, d, distance)

% Defaults
if nargin < 4 || isempty(distance)
    distance = 'all';
end

% Which estimates
read_euclid = ismember(distance, {'euclid', 'all'});
read_mahal  = ismember(distance, {'mahal' , 'all'});

% Initialize
if read_euclid
    x_best_euclid = NaN(n, d);
    f_best_euclid = NaN(n, 1);
    hess_euclid   = cell(1,n);
end
if read_mahal
    x_best_mahal  = NaN(n, d);
    f_best_mahal  = NaN(n, 1);
    hess_mahal    = cell(1,n);
end

% Load from file
for i = 1:n
    % Load file
    tmp = load( sprintf('%s%d', names, i) );
    
    % Euclid estimates
    if read_euclid
        x_best_euclid(i,:) = tmp.x_best_euclid;
        f_best_euclid(i,:) = tmp.f_best_euclid;
        hess_euclid{i}     = tmp.hess_euclid;
    end
    
    % Mahalanobis estimates
    if read_mahal
        x_best_mahal(i,:) = tmp.x_best_mahal;
        f_best_mahal(i,:) = tmp.f_best_mahal;
        hess_mahal{i}     = tmp.hess_mahal;
    end
end

% Create experiment structure
if read_euclid
    experiment.x_best_euclid = x_best_euclid;
    experiment.f_best_euclid = f_best_euclid;
    experiment.hess_euclid   = hess_euclid;
end
if read_mahal
    experiment.x_best_mahal  = x_best_mahal;
    experiment.f_best_mahal  = f_best_mahal;
    experiment.hess_mahal    = hess_mahal;
end

end