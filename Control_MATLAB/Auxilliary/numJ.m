function [ J ] = numJ(f,x0,dx)
%NUMJ Constructs the Jacobian of function f about x0 numerically    
%   The center-difference is used and dx represents the numerical step.

%% Important:
%   This and the Newton Iterations may be replaced by fzero() in the
%   future.

%% Initialization
nx0 = length(x0);                                                           %Number of variables w.r.t. to which check variation
nf = length(f(x0));                                                         %Number of rows of Jacobian
J = zeros(nf,nx0);                                                          %Empty Jacobian

%% Main Loop
for k = 1:nx0                                                               %Construct Jacobian with center difference
    e_k = zeros(nx0,1);
    e_k(k) = dx;
    J(:,k) = (f(x0+e_k) - f(x0-e_k))/(2*dx);                                
end

end