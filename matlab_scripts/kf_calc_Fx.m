%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% F = kf_calcDFx(x) Calculates the Jacobian of the system dynamics equation f(x,u,t) 
%   
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function DFx = kf_calcDFx(t, x, u)
    
    n = size(x, 1);
    
    % Calculate Jacobian matrix of system dynamics
    DFx = zeros(4);
end
    

