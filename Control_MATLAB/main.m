%% 0. Load System, Trajectories, and choose Energy to control to
%addpath(genpath('../encont-master/Shooting_Methods/Auxiliary_Functions'))
addpath(genpath('./Auxilliary'));
addpath(genpath('./DP_k_10'));

Y = load('Trajectories_DP_k_10.mat').Y;

E = 8;                                                                      % Desired energy 
k = 1;                                                                      % Generator (k=1 is more interesting)

dE = 1e+16;
i_center = 1;                                                               % Index of trajectory closes to desired energy
for i = 1:length(Y{k,2})                                                    % Pick closest trajectory
    E_Curr = Y{k,2}{i}(1,3);
    if abs(E_Curr - E) < dE
        dE = abs(E_Curr - E);
        i_center = i;
    end
end
E = Y{k,2}{i_center}(1,3);    

%% 1. Create distance function f with minimum on trajectory
% 1.0 Create q(t) and p(t). Here, p(t) is the velocity dq/dt ... 
n = 60;
t = Y{k,3}{i_center};
T = t(end);
q = Y{k,1}{i_center}(:,1:2);
p = Y{k,1}{i_center}(:,3:4);
t = t(t<=T/2)';
q = q(t<=T/2,:)';
p = p(t<=T/2,:)';

q_t = hermifit(q,t,n);
p_t = hermifit(p,t,n);  

% 1.1 Use distance from curve for f, i.e. f = @(q) min_d_q(q_t,q)
f = @(q) min_d_q( @(t) q_t(t*T/2) , q );
fdir = @(q) dir_min_d_q(@(t) q_t(t*T/2), q);


%% 2. Split kinetic energy into desired and undesired components (Happens in control section , now)
% Requirements for desired component: 
% - df/dq*dqdt < 0 with equality only at goal trajectory.
% - loose directional requirement, sign dependent on dqdt

% Design Choice required for desired component. Here: "parallel transport" of
% velocity at closest point of goal trajectory 


%% 3. Write control that routes energy from f+E_Kin_udes to V+E_Kin_des
% Strong Design Choice, could render performance independent of previous choices 

E_des = E;
F = @(u) force_in(u,q_t,p_t,E_des,T);                                       % Control input, is aware current state u and desired trajectory q_t,p_t
% see definition of force_in at bottom of this file

%% 4. Compute dynamics
Err_min = 1e-1;
opts = odeset('RelTol',Err_min*1e-3,'AbsTol',Err_min*1e-3);                 % Tolerance of ODE Solver
u0 = rand(4,1); %[q_t(0);p_t(0)]+0.1;%                                      % Initial state
ufinal = ode45(@(t,u)EoM_Control(u,T,F),[0,10],u0,opts);                    % Output of simulation

%% 4.1 Cycle Multipliers (Smaller than 1 means locally attracting orbit)
u0 = [q_t(T/4);p_t(T/4)];
psi_T = @(u0) deval(ode45(@(t,u)EoM_Control(u,T,F),[0,T],u0,opts),T);
Jac = numJ(psi_T,u0,1e-2);
CycleMultipliers = eig(Jac);

%% 5. Relevant plots 
% Plot Distance function:
nX = 100;
nY = 90;
qx = linspace(-1.5,1,nX);
qy = linspace(0,2.5,nY);
val = [];
for i = 1:nX
    for j = 1:nY
        val(i,j) = f([qx(i);qy(j)])/0.1;
    end
end
figure()
[QX,QY] = meshgrid(qx,qy);
contourf(QX,QY,val',20)
%% Plot target and actual trajectory
hold on
plot(q(1,:),q(2,:),'linewidth',2, 'Color','#A2142F')                        % red, target trajectory
plot(ufinal.y(1,:),ufinal.y(2,:),'--','linewidth',2, 'Color','#77AC30')     % green & dashed actual trajectory
hold off

%% Energy along actual trajectory
u_sim = ufinal.y;
[~, s2] =size(u_sim);
E_val = [];
for i = 1:s2
    E_val(i) = E_Total(u_sim(:,i),0);
end
figure()
plot(ufinal.x,E_val,'LineWidth',2,'Color','#77AC30');
xlabel('Time in s')
ylabel('Energy in J')

%% Distance & Log-Distance from target trajectory over time
Dist_Mode = [];
Log_Dist_Mode = [];
for i = 1:s2
    Dist_Mode(i) = min_d_q(@(t)q_t(t*T/2),u_sim(1:2,i));
    Log_Dist_Mode(i) =log(Dist_Mode(i));
end
figure()
plot(ufinal.x,Dist_Mode,'LineWidth',2,'Color','#77AC30');
xlabel('Time in s')
ylabel('Distance in rad')

figure()
plot(ufinal.x,Log_Dist_Mode,'LineWidth',2,'Color','#77AC30');
xlabel('Time in s')
ylabel('Log of Distance in rad')


%% Stabilizing Control
function [F] = force_in(u,q_t,p_t,E_des,T)
a_E = 1; % damping injection gain 
a_M = 10; % eigenmanifold control gain
unorm = sqrt(u(3:4).'*Mass(u)*u(3:4)); %norm(u(3:4));

% EKin = E_Kin(u);
% 
% dir_mode = dir_min_d_q(q_t,u(1:2));
% dm = norm(dir_mode);

if unorm>0                                                                  % unit vector along current velocity
    dqdt_hat = u(3:4)/unorm;
else 
    dqdt_hat = u(3:4);
end

dqdt_fin = p_t(T/2*t_min_d_q(@(t) q_t(t*T/2),u(1:2)));                      % velocity of closest point on desired trajectory 
Sig = sign(dqdt_fin.'*u(3:4));                                              % Sign depends on current system velocity
dqdt_des_raw = -Sig*dqdt_fin;                                               % desired velocity component

B = reduce_base(dqdt_hat,eye(2),Mass(u));                                   % basis components that are not in direction of system velocity
[~,sB] = size(B);

dqdt_des = zeros(2,1);
for i=1:sB
   dqdt_des = dqdt_des + (B(:,i).'*Mass(u)*(dqdt_des_raw))*B(:,i);          % desired adapted velocity s.t. F_Dir does not inject energy
end
F_M = -a_M*Mass(u)*dqdt_des;

F_E = - a_E*(E_Total(u,E_des))*Mass(u)*dqdt_hat;                            % Energy injection control
F = F_E + F_M;
end
