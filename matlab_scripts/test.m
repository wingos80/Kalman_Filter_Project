%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of the Linear Kalman Filter
%
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
close all;
clear all;
randn('seed', 7);

load_f16data2023
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set simulation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n               = 4;            % number of states
nm              = 3;            % number of measurements
m               = 3;            % number of inputs
dt              = 0.01;         % time step (s)
N               = length(U_k);  % sample size
epsilon         = 1e-10;
doIEKF          = 1;            % set 1 for IEKF and 0 for EKF
maxIterations   = 100;

printfigs = 0;
figpath = '';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set initial values for states and statistics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
E_x_0       = [150; 0; 0; -0.6];     % initial estimate of optimal value of x_k1_k1

B           = [1, 0, 0; 
               0, 1, 0;
               0, 0, 1;
               0, 0, 0];                     % input matrix
G           = [0, 0, 0, 0; 
               0, 0, 0, 0;
               0, 0, 0, 0;
               0, 0, 0, 0];                     % noise input matrix

% Initial estimate for covariance matrix
std_x_0     = 1;
P_0         = [std_x_0^2, 0, 0, 0; 
               0, std_x_0^2, 0, 0;
               0, 0, std_x_0^2, 0;
               0, 0, 0, std_x_0^2];

% System noise statistics:
E_w         = 0;                            % bias of system noise
std_w       = 1e-3;                         % standard deviation of system noise
Q           = [std_w^2, 0, 0, 0; 
               0, std_w^2, 0, 0;
               0, 0, std_w^2, 0;
               0, 0, 0, 0];                 % variance of system noise

% Measurement noise statistics:
E_v         = 0;                            % bias of measurement noise
std_nu_a    = 0.035;                        % standard deviation of alpha noise
std_nu_b    = 0.010;                        % standard deviation of beta noise
std_nu_V    = 0.110;                        % standard deviation of velocity noise


R      = [std_nu_a^2, 0, 0;
          0, std_nu_b^2, 0;
          0, 0, std_nu_V^2];                % variance of system noise

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize Iterated Extended Kalman filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t_k             = 0; 
t_k1            = dt;

XX_k1_k1        = zeros(n, N);
PP_k1_k1        = zeros(n, N);
STD_x_cor       = zeros(n, N);
STD_z           = zeros(nm, N);
ZZ_pred         = zeros(nm, N);
IEKFitcount     = zeros(N, 1);

x_k1_k1         = E_x_0;    % x(0|0)=E{x_0}
P_k1_k1         = P_0;      % P(0|0)=P(0)

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run the Extended Kalman filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

% Run the filter through all N samples
for k = 1:N
    % x(k+1|k) (prediction)
    [t, x_k1_k]     = rk4(@kf_calc_f, x_k1_k1,U_k(:,k), [t_k t_k1]); 

    % z(k+1|k) (predicted observation)
    z_k1_k          = kf_calc_h(0, x_k1_k, U_k(k));     % prediction of observation
%     ZZ_pred(:,k)
    ZZ_pred(:,k)      = z_k1_k;   % store this observation prediction, since later prediction observations
                                % are corrected using the actual observation
        
    % Calc Jacobians Phi(k+1,k) and Gamma(k+1, k)
    Fx              = kf_calc_Fx(0, x_k1_k, U_k(:,k)); % perturbation of f(x,u,t)
    % the continuous to discrete time transformation of Df(x,u,t) and G
    [dummy, Psi]    = c2d(Fx, B, dt);   
    [Phi, Gamma]    = c2d(Fx, G, dt);   


    % P(k+1|k) (prediction covariance matrix)
    P_k1_k          = Phi*P_k1_k1*Phi.' + Gamma*Q*Gamma.';
    
    % Run the Iterated Extended Kalman filter (if doIEKF = 1), else run standard EKF
    if (doIEKF)
        % do the iterative part
        eta2    = x_k1_k;
        err     = 2*epsilon;

        itts    = 0;
        while (err > epsilon)
            if (itts >= maxIterations)
                fprintf('Terminating IEKF: exceeded max iterations (%d)\n', maxIterations);
                break
            end
            itts    = itts + 1;
            eta1    = eta2;

            % Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix 
            Hx       = kf_calc_Hx(0, eta1, U_k(:,k)); 
            
            % Check observability of state
%             if (k == 1 && itts == 1)
%                 rankHF  = kf_calcObsRank(Hx, Fx);
%                 if (rankHF < n)
%                     warning('The current state is not observable; rank of Observability Matrix is %d, should be %d', rankHF, n);
%                 end
%             end
         
            % Observation and observation error predictions
            z_k1_k      = kf_calc_h(0, x_k1_k, U_k(k));     % prediction of observation 
            P_zz        = (Hx*P_k1_k*Hx.' + R);             % covariance matrix of observation error
            std_z       = sqrt(diag(P_zz));                 % standard deviation of observation error (for validation) 

            % calculate the Kalman gain matrix
            K           = P_k1_k*Hx'/P_zz;
            % new observation state
            z_p         = kf_calc_h(0, eta1, U_k(:,k));

            eta2        = x_k1_k + K*(Z_k(:,k) - z_p - Hx*(x_k1_k - eta1));
            err         = norm((eta2 - eta1), inf)/norm(eta1, inf);
        end

        IEKFitcount(k)  = itts;
        x_k1_k1         = eta2;
    else
        % Correction
        Hx              = kf_calc_Hx(0, x_k1_k, U_k(:,k));  % perturbation of h(x,u,t)
        
        % P_zz(k+1|k) (covariance matrix of innovation)
        P_zz            = (Hx*P_k1_k * Hx.' + R);           % covariance matrix of observation error
        std_z           = sqrt(diag(P_zz));                 % standard deviation of observation error (for validation)    

        % K(k+1) (gain)
        K               = P_k1_k * Hx.'/P_zz;
        
        % Calculate optimal state x(k+1|k+1) 
        x_k1_k1         = x_k1_k + K*(Z_k(:,k) - z_k1_k); 
    end    
   
    P_k1_k1         = (eye(n) - K*Hx) * P_k1_k * (eye(n) - K*Hx).' + K*R*K.';  
    std_x_cor       = sqrt(diag(P_k1_k1));

    % Next step
    t_k             = t_k1; 
    t_k1            = t_k1 + dt;
    
    % store results
    XX_k1_k1(:,k)     = x_k1_k1;
    PP_k1_k1(:,k)     = diag(P_k1_k1);
    STD_x_cor(:,k)    = std_x_cor;
    STD_z(:,k)        = std_z;
end

disp('complete')
plot(STD_x_cor(1,:))
