function [ F ] = Comp_Dyn(u,du)
%Comp_Dyn control force required to have double pendulum with springs and 
%gravity follow state u with change-rate du.
%   Output is the rate of change u' for system defined in EoM of type u' =
%   f(u,T). Here, u = [u_1 u_2 u_3 u_4] consists of configuration
%   variables u_1 to u_2 and velocities u_3 to u_4.

%% Initialization
% Parameters locally defined: Faster than sharing global workspace
 d1 = 1;
 d2 = 1;
 m1 = 0.4;
 m2 = 0.4;
 I1 = 0;
 I2 = 0;
 G = 9.81;
 K = 10; %8; %10
 M = zeros(2);                                                                % Pre-allocate space to large matrix

%% Calculating the constituents
% System of equations is M*x_tt = Mx_tt(x,x_t), in first order form.

 Mx_tt = -[G*(d1*m1*sin(u(1))+d1*m2*sin(u(1))+d2*m2*sin(u(1)+u(2)))-...
           2*d1*d2*m2*sin(u(2))*u(3)*u(4)-d1*d2*m2*sin(u(2))*u(4)^2;...
         -((K*pi)/2)+d2*G*m2*sin(u(1)+u(2))+K*u(2)+d1*d2*m2*sin(u(2))*u(3)^2];

% Mass matrix, using symmetry
 M(1,1)= I1+d1^2*m1+d1^2*m2+d2^2*m2+2*d1*d2*m2*cos(u(2));
 M(1,2)= d2*m2*(d2+d1*cos(u(2)));
 M(2,1)= M(1,2);... 
 M(2,2)= I2+d2^2*m2;

 %% Matrix Inverison and Output
 F = M*du(3:4) - Mx_tt;
end

