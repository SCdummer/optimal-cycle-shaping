function [x_fit] = bezifit(x,t,n)
%BEZIFIT for x(:,j) = x(t(j)) fit piecewise cubic bezier curve using roughly n points 
%   Uses GnBezierFit. 
%   Output x_fit(t) is a function of t, with x_fit(t(j)) approximately x(:,j). 
%   n_max = size(R,2)

%% Initial checks
N = size(x,1);                                                              % Length of individual y
n_max = size(x,2)-1;                                                        % Maximum feasible number of iterations given measurements in R, which needs at least i+2*(n-1)-1
if n > n_max                                                                % Limiting Maximum Accuracy if required
    n = n_max; 
    fprintf('Accuracy in bezifit.m limited to %i for lack of measurements',n);
end

%% Downsample t and x
t_red = [downsample(t,floor(n_max/n))];
if t_red(end)~= t(end)
    t_red(end+1) = t(end);
end
[~,ind] = find(~(t-t_red'));
x_red = x(:,ind);

%% Construct coordinate-wise bezier approximation 
% t = t - t(end);                                                           % Make sure that t = 0 corresponds to last point
x_fit = @(dt)[];                                                            % Handle for full approximation

for i = 1:N                                                                 % Iterate over N coordinates
    out = GnBezierFit([t_red;x_red(i,:)],3);                                % Fit piecewise cubic bezier curve 
    nc = length(out);                                                       % Number of curves
    y = @(t)0;
    for j = 1:nc                                                            % Iterate over Segments
        y = @(t) y(t)+(t>=(j-1)/nc)*(t<(j/nc))*out(j).Q*bernsteinMatrix(4,mod(t*nc,1))';
    end
    x_fit = @(dt) [x_fit(dt); y(dt)];                                       
end


end



