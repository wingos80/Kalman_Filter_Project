%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Testing observability of the system
%
%   Author: W. Chan, Delft University of Technology, 2023
%   email: w.y.chan-1@student.tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
close all;
clear all;

% define variables
syms('u', 'v', 'w', 'Caup', 'am', 'bm', 'vm', 'udot', 'vdot', 'wdot');

% define state vector
x  = [ u;  v;  w;  Caup ];
x_0 = [ 2; 3;  5;  7]; % initial state, random numbers

nstates = length(x); % length of state vector

% define state transition function
f = [udot;
     vdot;
     wdot;
     0];
 
% define state observation function
h = [atan(w/u)*(1+Caup);
     atan(v/sqrt(u^2+w^2));
     sqrt(u^2+v^2+w^2)];

% Compute rank of the observability matrix
rank = kf_calcNonlinObsRank(f, h, x, x_0);
