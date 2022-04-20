function [ EPot ] = V( u, E )
%POTENTIAL Potential function using E as the reference for the 0-Level
%   V(u,0) is the potential energy of the double pendulum with joint
%   springs and gravity. Further, V(u,E) = V(u,0) - E;

%% Initialization
 d1 = 1;
 d2 = 1;
 m1 = 0.4;
 m2 = 0.4;
 G = 9.81;
 K = 0; %15; %8; 
 ueq =  [0;0]; %[-0.4165;1.3592]; %[-0.3781;1.2086]; %
 
%% Main Calculations
 EPot = 1/2*K*(u(2)-pi/2)^2 -d1*m1*G*cos(u(1))-(d1*cos(u(1))+d2*cos(u(1)+u(2)))*m2*G - E; % Equal to 0 at x corresponding to potential energy E w.r.t. global minimum
 EPot0 = 1/2*K*(ueq(2)-pi/2)^2 -d1*m1*G*cos(ueq(1))-(d1*cos(ueq(1))+d2*cos(ueq(1)+ueq(2)))*m2*G;
 EPot = EPot - EPot0;
 end
