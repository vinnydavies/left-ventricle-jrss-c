% Add everything to path
addpath( genpath(pwd) )


% Required toolboxes in the path:
%   AdaptiveRobustNumericalDifferentiation (John D'Errico)
%   gpml-matlab-2007 (Carl Edward Rasmussen and Hannes Nickisch)
%   varsgpv1.0 (Michalis K. Titsias)

% Numerical gradient estimation
addpath( genpath(fullfile('toolboxes','AdaptiveRobustNumericalDifferentiation')) )

% GPML
addpath( genpath(fullfile('toolboxes','gpml-matlab-2007')) )

% Variational Sparse GPs by Titsias
addpath( genpath(fullfile('toolboxes','varsgpv1.0')) )
