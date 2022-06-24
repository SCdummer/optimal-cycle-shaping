function [x_fit] = hermifit(x,t,n)
%HERMIFIT for x(:,j) = x(t(j)) fit piecewise cubic hermite polynomials using roughly n points 
%   Uses makima, pchip is an aggressive side option 
%   Output x_fit(t) is a function of t, with x_fit(t(j)) approximately x(:,j). 
%   n_max = size(R,2)

%% Initial checks
N = size(x,1);                                                              % Length of individual y
n_max = size(x,2)-1;                                                        % Maximum feasible number of iterations given measurements in R, which needs at least i+2*(n-1)-1
if n > n_max                                                                % Limiting Maximum Accuracy if required
    n = n_max; 
    fprintf('Accuracy in hermifit.m limited to %i for lack of measurements',n);
end

%% Downsample t and x
t_red = [downsample(t,floor(n_max/n))];
if t_red(end)~= t(end)
    t_red(end+1) = t(end);
end
[~,ind] = find(~(t-t_red'));
x_red = x(:,ind);

%% Construct coordinate-wise hermite approximation 
% t = t - t(end);                                                           % Make sure that t = 0 corresponds to last point
x_fit = @(dt)[];                                                            % Handle for full approximation

out = makima(t_red,x_red);                                                  % Fit piecewise cubic hermite curve 
x_fit = @(t) ppval(out,t);      

end




