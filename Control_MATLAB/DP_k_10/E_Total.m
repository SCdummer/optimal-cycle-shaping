function [ ETotal ] = E_Total( u, E )
%POTENTIAL Potential function using E as the reference for the 0-Level
%   V(u,0) is the potential energy of the double pendulum with joint
%   springs and gravity. Further, V(u,E) = V(u,0) - E;

ETotal = V(u,E) + E_Kin(u);
 end
