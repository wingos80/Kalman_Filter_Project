%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of the symbolic construction of the state observation matrix
%
%   Author: Wing Chan, adapted from C.C. de Visser, Delft University of Technology, 2013
%   email: wingyc80@gmail.com, c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
close all;
clear all;

% define variables
syms('g','x','y','z','u','v','w','phi','theta','psi','wxe','wye','wze','lx','ly','lz','lp','lq','lr','ax','ay','az','p','q','r','V','alpha','beta');
% define state vector
X  = [ x; y; z; u; v; w; phi; theta; psi; wxe; wye; wze; lx; ly; lz; lp; lq; lr];
X0  = [ 1; 2; 3; 5; 7; 11; 13; 17; 19; 23; 29; 31; 37; 41; 1.1; 1.2; 1.4; 1.5];

% define state transition function
f = [(u*cos(theta) + (v*sin(phi)+w*cos(phi))*sin(theta))*cos(psi) - sin(psi)*(v*cos(phi) - w*sin(phi)) + wxe; 
     (u*cos(theta) + (v*sin(phi)+w*cos(phi))*sin(theta))*sin(psi) + cos(psi)*(v*cos(phi) - w*sin(phi)) + wye;  
     -u*sin(theta) + cos(theta)*(v*sin(phi) + w*cos(phi)) + wze;
     (ax-lx) - g*sin(theta) + (r-lr)*v - (q-lq)*w;
     (ay-ly) + g*cos(theta)*sin(phi) + (p-lp)*w - (r-lr)*u;
     (az-lz) + g*cos(theta)*cos(phi) + (q-lq)*u - (p-lp)*v;
     (p-lp) + (q-lq)*sin(phi)*tan(theta) + (r-lr)*cos(phi)*tan(theta);
     (q-lq)*cos(phi) - (r-lr)*sin(phi);
     (q-lq)*sin(phi)/cos(theta) + (r-lr)*cos(phi)/cos(theta);
     0;
     0;
     0;
     0;
     0;
     0;
     0;
     0;
     0];

 
% define state observation function
h = [x;
     y;
     z;
     (u*cos(theta) + (v*sin(phi)+w*cos(phi))*sin(theta))*cos(psi) - sin(psi)*(v*cos(phi) - w*sin(phi)) + wxe; 
     (u*cos(theta) + (v*sin(phi)+w*cos(phi))*sin(theta))*sin(psi) + cos(psi)*(v*cos(phi) - w*sin(phi)) + wye;  
     -u*sin(theta) + cos(theta)*(v*sin(phi) + w*cos(phi)) + wze;
     phi;
     theta;
     psi;
     sqrt(u^2+v^2+w^2);
     atan2(w,u);
     atan2(v,sqrt(u^2+w^2))];

rank = kf_calcNonlinObsRank(f, h, X, X0)


function rankObs = kf_calcNonlinObsRank(f, h, X, X0)

nstates         = length(X);
nobs            = length(h);

Hx              = simplify(jacobian(h, X));
ObsMat          = zeros(nobs*nstates, nstates);
Obs             = sym(ObsMat, 'r');
Obs(1:nobs, :)  = Hx;
Obsnum          = subs(Obs, X, X0);
rankObs         = double(rank(Obsnum));
fprintf('\nRank of Initial Observability matrix is %d\n', rankObs);
if (rankObs >= nstates)
    fprintf('Observability matrix is of Full Rank: the state is Observable!\n');
    return;
end
LfHx    = simplify(Hx * f);
for i = 2:nstates
    LfHx                        = jacobian(LfHx, X);
    Obs((i-1)*nobs+1:i*nobs,:)  = LfHx;
    Obsnum                      = subs(Obs, X, X0);
    rankObs                     = double(rank(Obsnum));
    fprintf('\t-> Rank of Observability matrix is %d\n', rankObs);
    if (rankObs >= nstates)
        fprintf('Observability matrix is of Full Rank: the state is Observable!\n');
        return;
    end
    LfHx    = (LfHx * f);
    fprintf('Loop %d took %2.2f seconds to complete\n', i, toc);
end
fprintf('WARNING: Rank of Observability matrix is %d: the state is NOT OBSERVABLE!\n', rankObs);

end

