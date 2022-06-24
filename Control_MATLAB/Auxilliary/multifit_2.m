function [x_fit] = multifit_2(x,t,n)
%MULTIFIT_2 for x(:,j) = x(t(j)) fit n-th order polynomial
%   Uses polyfit. 
%   Output x_fit(t) is a function of t, with x_fit(t(j)) approximately x(:,j). 
%   n_max = size(R,2)

N = size(x,1);                                                              % Length of individual y
n_max = size(x,2)-1;                                                        % Maximum feasible number of iterations given measurements in R, which needs at least i+2*(n-1)-1
if n > n_max                                                                % Limiting Maximum Accuracy if required
    n = n_max; 
    fprintf('Accuracy in predict.m limited to %i for lack of measurements',n);
end

%% Construct coordinate-wise polynomial approximation 
%t = t - t(end);                                                             % Make sure that t = 0 corresponds to last point
x_fit = @(dt)[];                                                            % Handle for full approximation

for i = 1:N
    [p,~,MU] = polyfit(t,x(i,:),n);                                              % Fit polynomial
    x_fit = @(dt) [x_fit(dt); polyval(p,dt,[],MU)];                         % Iterate over N coordinates 
end


end



