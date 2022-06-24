function [t] = t_of_q(q_t,q,t_max)
% invert q(t) to t(q)

Err_min = 10^-4;                                                            % Minimum Error required
nMax = 4;
opts_fsolve = optimoptions(@fsolve,...
    'FunctionTolerance', Err_min*1e-2,'OptimalityTolerance', Err_min*1e-2, ... % Options of fsolve: levenberg-marquardt or trust-region
    'Algorithm', 'trust-region','Display','off','MaxIterations',nMax);

% distance function with unique minimum
dist = @(q,t) vecnorm(q_t(t) - q*ones(1,length(t)));

% construct corse first guess
ngrid = 10;
t_grid = linspace(0,t_max,ngrid);
[~,ind] = min(dist(q,t_grid));

%refine guess
t = fsolve(@(t) dist(q,t),t_grid(ind),opts_fsolve);
end