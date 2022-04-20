function [x] = fKin(theta)
%FKIN Computes CoM positions x_1 to x_DoF from joint angles theta
%   This version holds for the double Pendulum
%% Parameters
d = 1;

%% Conversion
x = zeros(4,1);
x(1:2) = d*[sin(theta(1)); -cos(theta(1))];
x(3:4) = x(1:2) + d*[sin(theta(1)+theta(2)); -cos(theta(1)+theta(2))];

end

