function [logP, P] = Probit(mu)
%

P = (1+erf(mu/sqrt(2)))/2;  
logP = zeros(size(mu));
b =  0.158482605320942;           % quadratic asymptotics approximated at -6
c = -1.785873318175113;    
ok = mu>-6;                            % normal evaluation for larger values
logP(ok) = log(P(ok)); 
logP(~ok) = -mu(~ok).^2/2 + b*mu(~ok) + c;                  % log of sigmoid
