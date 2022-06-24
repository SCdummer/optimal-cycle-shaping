function [q_dir] = dir_min_d_q(q_t,qc)
%DIR_MIN_D_Q direction from qc to closest point on line q_t
Err_min = 1e-4;                                                            % Minimum Error required
nMax = 10;

% opts_fsolve = optimoptions(@fsolve,...
%    'FunctionTolerance', Err_min*1e-2,'OptimalityTolerance', Err_min*1e-2, ... % Options of fsolve: levenberg-marquardt or trust-region
%    'Algorithm', 'trust-region','Display','off','MaxIterations',nMax);

% opts_lsqnonlin = optimoptions(@lsqnonlin,...
%    'FunctionTolerance', Err_min*1e-2,'OptimalityTolerance', Err_min*1e-2, ... % Options of lsnonlin
%    'Algorithm', 'trust-region-reflective','Display','off','MaxIterations',nMax);

% distance function with unique minimum

dist = @(q,t) vecnorm(q_t(t) - q*ones(1,length(t)));
dist2 = @(q,t) abs(q_t(t)-q*ones(1,length(t)));

% construct corse first guess
ngrid = 50;
t_grid = linspace(0,1,ngrid);
[~,ind] = min(dist(qc,t_grid));
t = t_grid(ind);

%refine guess
% 
% [t] = fsolve(@(t) dist(qc,t),t_grid(ind),opts_fsolve); %lsqnonlin is an option
% t = lsqnonlin(@(t) dist2(qc,t), t_grid(ind),0,1,opts_lsqnonlin);
% qm = q_t(t);
% dm = norm(qm-qc,2);

% fixed refining step: after corse guess, approximate further by polynomial
q = q_t(t);
dq = numJ(q_t,t,Err_min);
ddq = numJ(@(x) numJ(q_t,x,Err_min), t, Err_min);
p = [2*ddq.^2, 3*ddq.*dq, (2*ddq.*(q-qc)+dq.^2),dq.*(q-qc)];
r = roots(p(1,:)+p(2,:));
dm = inf;
for i = 1:3
    qm = q_t(t+real(r(i)));
    dm_new = norm(qm-qc,2);
    if dm_new < dm
        dm = dm_new;
        dt = real(r(i));
    end
end
t = t+dt;
q_dir = (q_t(t)-qc);

end

