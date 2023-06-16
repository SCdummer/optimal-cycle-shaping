import torch
import torch.nn as nn
from torch.autograd import grad as grad
import numpy as np
from src.opt_limit_cycle_control.utils import numJ, cuberoot


# parameters pendulum
m, k, l, qr, b, g = 1., 0.5, 1, 0, 0.01, 9.81

# parameters double pendulum
m1, m2, k1, k2, l1, l2, qr1, qr2, b1, b2 = 1., 1., 0.5, 0.5, 1, 1, 0, 0, 0.01, 0.01


class ControlledSystemDoublePendulum(nn.Module):
    # Elastic Double Pendulum Model
    def __init__(self, V, T_initial=1.0, T_requires_grad=True):
        super().__init__()
        if T_requires_grad:
            self.V, self.T, self.n = V, torch.nn.Parameter(torch.tensor([T_initial])), 2
        else:
            self.V, self.T, self.n = V, torch.tensor([T_initial]), 2
            self.T.requires_grad = False

    def forward(self, t, x, V_only=False):
        if V_only:
            q = x[...].view(-1, self.n)
            return torch.cat([self._potential_shaping(q), self._energy_shaping(q.requires_grad_(True))], dim=1)
        else:
            # Evaluates the closed-loop vector field
            with torch.set_grad_enabled(True):
                q1, q2, p1, p2 = x[..., 0].view(-1, 1), x[..., 1].view(-1, 1), x[..., 2].view(-1, 1), x[..., 3].view(-1,
                                                                                                                     1)
                q1 = q1.requires_grad_(True)
                q2 = q2.requires_grad_(True)
                # compute control action
                q = torch.cat([q1, q2], dim=1)
                p = torch.cat([p1, p2], dim=1)
                u = self._energy_shaping(q)
                # compute dynamics
                dxdt = torch.abs(self.T[0]) * self._dynamics(q, p, u)

        return dxdt

    def _dynamics(self, q, p, u):
        u1 = u[:,0].unsqueeze(-1)
        u2 = u[:,1].unsqueeze(-1)
        q1 = q[:,0].unsqueeze(-1)
        q2 = q[:,1].unsqueeze(-1)
        p1 = p[:,0].unsqueeze(-1)
        p2 = p[:,1].unsqueeze(-1)

        DET = (l1**2 * l2**2 * m2**2 * (-torch.cos(q2)**2) + l1**2 * l2**2 * m2**2 + l1**2 * l2**2 * m1 * m2)

        dq1dt = (l2**2 * m2 * p1) / DET - (l2 * m2 * p2 * (l1 * torch.cos(q2) + l2)) / DET
        dq2dt = (p2 * (2 * l2 * l1 * m2 * torch.cos(q2) + l1**2 * (m1 + m2) + l2**2 * m2))/ DET - (l2 * m2 * p1 * (l1 * torch.cos(q2) + l2))/ DET

        dp1dt = (-g) * (torch.sin(q1 + q2) * l2 * m2 + torch.sin(q1) * l1 * (m1 + m2)) + u1
        dp2dt = -((1/(l1**2 * l2**2 * (m1+torch.sin(q2)**2 * m2)**2)) * ((-p2**2) * torch.sin(q2) * (torch.cos(q2)*l1+l2) * (torch.cos(q2)*l2*m2+l1*(m1+m2))\
        + (1/2)* p1 * p2 * torch.sin(q2) * l2 * (4*torch.cos(q2)*l2*m2+l1*(2*m1+(3+torch.cos(2*q2))*m2))\
        + l2**2 * ((-torch.cos(q2)) * p1**2 * torch.sin(q2) * m2 + l1**2 * (m1+torch.sin(q2)**2 * m2)**2 *(g*torch.sin(q1+q2)*l2*m2-k2*(torch.pi-2*q2)))))\
        + u2
        

        return torch.cat([dq1dt, dq2dt, dp1dt, dp2dt], dim=1)

    def _energy_shaping(self, q):
        # energy shaping control action
        dVdx = grad(self.V(q).sum(), q, create_graph=True)[0]
        #dVdx = grad(self.V(q).sum(), q, create_graph=True)[0]
        return -dVdx

    def _potential_shaping(self, q):
        # energy shaping control action
        V = self.V(q)
        return V

    def _autonomous_energy(self, q, p):
        # Hamiltonian (total energy) of the UNCONTROLLED system
        q1 = q[:,0].unsqueeze(-1)
        q2 = q[:,1].unsqueeze(-1)
        Grav_Pot = -g*(torch.cos(q1 + q2)*l2*m2 + torch.cos(q1)*l1*(m1 + m2))
        Spring_pot = k2*(1/2*torch.pi - q2)**2
        Kin_E = 1/2*torch.inner(p.squeeze(),self._inv_mass_tensor(q)@p.squeeze())
        return Grav_Pot + Kin_E + Spring_pot, Grav_Pot, Kin_E, Spring_pot

    def _energy(self, q, p):
        # Hamiltonian (total energy) of the CONTROLLED system
        return self._autonomous_energy(q,p)+self._potential_shaping(q).sum()


class AugmentedDynamicsDoublePendulum(nn.Module):
    # "augmented" vector field to take into account integral loss functions
    def __init__(self, f, int_loss):
        super().__init__()
        self.f = f
        self.int_loss = int_loss
        self.nfe = 0.

    def forward(self, t, x):
        self.nfe += 1
        x = x[:,:4]
        dxdt = self.f(t, x, V_only=False)
        dldt = self.int_loss(t, x).view(-1, 1)
        return torch.cat([dxdt, dldt], dim=1)


class StabilizedSystemDoublePendulum(nn.Module):
    # Elastic Double Pendulum Model
    def __init__(self, V, a_M=torch.tensor(0.0), a_E=torch.tensor(0.0),x_fun = lambda t: torch.tensor((0, 0, 0, 0)).float(), x_num = torch.tensor(((0, 0, 0, 0),(0, 0, 0, 0))).float() ):
        super().__init__()
        self.V, self.T, self.n = V, torch.nn.Parameter(torch.tensor([1.0])), 2
        self._set_control_parameters(a_E, a_M, x_fun, x_num) # Mode control- and energy-injection gains, desired mode 
        self.b = 0 # damping coefficient

    def forward(self, t, x, V_only=False):
        if V_only:
            q = x[...].view(-1, self.n)
            return torch.cat([self._potential_shaping(q), self._energy_shaping(q.requires_grad_(True))],dim=1)
        else:
            # Evaluates the closed-loop vector field
            with torch.set_grad_enabled(True):
                q1, q2, p1, p2 = x[..., 0].view(-1, 1), x[..., 1].view(-1, 1), x[..., 2].view(-1, 1), x[..., 3].view(-1, 1)
                q1 = q1.requires_grad_(True)
                q2 = q2.requires_grad_(True)
                # compute control action
                q = torch.cat([q1, q2], dim=1)
                p = torch.cat([p1, p2], dim=1)
                u = self._energy_shaping(q) + self._stabilizing_control(q,p)
                # compute dynamics
                dxdt = torch.abs(self.T[0]) * self._dynamics(q, p, u)

        return dxdt

    def _dynamics(self, q, p, u):
        u1 = u[:,0].unsqueeze(-1)
        u2 = u[:,1].unsqueeze(-1)
        q1 = q[:,0].unsqueeze(-1)
        q2 = q[:,1].unsqueeze(-1)
        p1 = p[:,0].unsqueeze(-1)
        p2 = p[:,1].unsqueeze(-1)
        


        # NEW controlled elastic pendulum dynamics: (one spring, no damping)
        DET = (l1**2 * l2**2 * m2**2 * (-torch.cos(q2)**2) + l1**2 * l2**2 * m2**2 + l1**2 * l2**2 * m1 * m2)

        dq1dt = (l2**2 * m2 * p1) / DET - (l2 * m2 * p2 * (l1 * torch.cos(q2) + l2)) / DET
        dq2dt = (p2 * (2 * l2 * l1 * m2 * torch.cos(q2) + l1**2 * (m1 + m2) + l2**2 * m2))/ DET - (l2 * m2 * p1 * (l1 * torch.cos(q2) + l2))/ DET

        dp1dt = (-g) * (torch.sin(q1 + q2) * l2 * m2 + torch.sin(q1) * l1 * (m1 + m2)) + u1 - self.b*dq1dt
        dp2dt = -((1/(l1**2 * l2**2 * (m1+torch.sin(q2)**2 * m2)**2)) * ((-p2**2) * torch.sin(q2) * (torch.cos(q2)*l1+l2) * (torch.cos(q2)*l2*m2+l1*(m1+m2))\
        + (1/2)* p1 * p2 * torch.sin(q2) * l2 * (4*torch.cos(q2)*l2*m2+l1*(2*m1+(3+torch.cos(2*q2))*m2))\
        + l2**2 * ((-torch.cos(q2)) * p1**2 * torch.sin(q2) * m2 + l1**2 * (m1+torch.sin(q2)**2 * m2)**2 *(g*torch.sin(q1+q2)*l2*m2-k1*(torch.pi-2*q2)))))\
        + u2 - self.b*dq2dt
        

        return torch.cat([dq1dt, dq2dt, dp1dt, dp2dt], dim=1)

    def _energy_shaping(self, q):
        # energy shaping control action
        dVdx = grad(self.V(q).sum(), q, create_graph=True)[0]
        return -dVdx

    def _potential_shaping(self, q):
        # energy shaping control action
        V = self.V(q)
        return V
    
    def _autonomous_potential(self,q):
        q1 = q[:,0].unsqueeze(-1)
        q2 = q[:,1].unsqueeze(-1)
        return -g*(torch.cos(q1 + q2)*l2*m2 + torch.cos(q1)*l1*(m1 + m2)) + k1*(1/2*torch.pi - q2)**2
        
    def _autonomous_kinetic(self,q, p):
        return 1/2*torch.inner(p.squeeze(),self._inv_mass_tensor(q)@p.squeeze())

    def _autonomous_energy(self, q, p):
        # Hamiltonian (total energy) of the UNCONTROLLED system
        q1 = q[:,0].unsqueeze(-1)
        q2 = q[:,1].unsqueeze(-1)
        Pot = -g*(torch.cos(q1 + q2)*l2*m2 + torch.cos(q1)*l1*(m1 + m2)) + k1*(1/2*torch.pi - q2)**2
        Kin = 1/2*torch.inner(p.squeeze(),self._inv_mass_tensor(q)@p.squeeze())
        return Pot+Kin

    def _energy(self, q, p):
        # Hamiltonian (total energy) of the CONTROLLED system
        return self._autonomous_energy(q,p)+self._potential_shaping(q).sum()
    
    def _stabilizing_control(self,q,p):
        # mode and energy stabilizing control action
        #self.q_t,self.p_t,self.E_Des,self.T
        
        pnorm = torch.sqrt(torch.dot(p.squeeze(),self._inv_mass_tensor(q)@p.squeeze()))
        if pnorm > 0:
            p_hat = p/pnorm;
        else:
            p_hat = p;
        
        p_fin = self._p_t(self._t_min_d_q(q));                           # momentum of closest point on desired trajectory 
        Sig = torch.sign(torch.dot(p_fin,p.squeeze()));                                  # Sign depends on current system momentum
        p_des_raw = Sig*p_fin;                                                 # desired momentum component
        p_des = p_des_raw - torch.dot(p_des_raw.squeeze(),self._inv_mass_tensor(q)@p_hat.squeeze())*p_hat;
        F_M = self.a_M*p_des;                                                  # Mode stabilizing control
        
        Diff_E = self.E_des - self._energy(q,p)
        F_E = self.a_E*Diff_E*p_hat;#*pnorm/(pnorm**2 + 1e-9);                       # Energy injection control
        F = F_E + F_M;
        
        return F

    def _q_t(self, t):
        return self.x_t(t).squeeze()[:2]

    def _p_t(self, t):
        return self.x_t(t).squeeze()[2:]

    def _t_min_d_q(self, qc):
        #time of minimum approach of trajectory self._q_(t) from qc
        len_xnum = torch.tensor(self.x_num).size()[0]
        i = torch.argmin(torch.norm(torch.tensor(self.x_num[:np.int(np.floor(len_xnum/2)),:2]).to(qc)-qc,dim=1))
        # i = torch.argmin(torch.norm(torch.tensor(self.x_num[:np.int(len_xnum/2),:2]).to(qc)-qc,dim=1))
        T_rough = self.T*i/torch.tensor(self.x_num[...,:2]).size()[0]
        Err_min = 1e-4
        q = self._q_t(T_rough);
        dq = numJ(self._q_t,T_rough,Err_min).squeeze();
        ddq = numJ(lambda x: numJ(self._q_t,x,Err_min).squeeze(), T_rough, Err_min).squeeze();
        p = torch.cat((2*torch.mul(ddq,ddq).unsqueeze(-1),
                       3*torch.mul(ddq,dq).unsqueeze(-1),
                       (2*torch.mul(ddq,(q-qc.squeeze()))+torch.mul(dq,dq)).unsqueeze(-1),
                       torch.mul(dq,(q-qc.squeeze())).unsqueeze(-1)),-1)
        p = p[0,:]+p[1,:];
        r = cuberoot(p) # used to return all roots in matlab
        dm = torch.norm(q-qc,dim=1);
        dt = torch.zeros(1).to(T_rough)
        #for i in range(1): # used to check all roots, best end-point of interval if none are better
        qm = self._q_t(T_rough+r);
        dm_new = torch.norm(qm-qc,dim=1);
        if dm_new[0] < dm[0]:
            dm = dm_new;
            dt = r;

        T_fine = T_rough+dt;

        return T_fine
    
    
    def _min_d(self,qc):
        return torch.norm(qc - self._q_t(self._t_min_d_q(qc)))
    
    def _q_des(self,q):
        # desired position at current point
        return self._q_t(self._t_min_d_q(q))
    
    def _p_des_raw(self,q,p):
        # desired final momentum at current point
        
        p_fin = self._p_t(self._t_min_d_q(q));                           # momentum of closest point on desired trajectory 
        Sig = torch.sign(torch.dot(p_fin,p.squeeze()));                                  # Sign depends on current system momentum
        return Sig*p_fin; 
    
    def _phase_dist(self,q,p):
        t = self._t_min_d_q(q)
        p_fin = self._p_t(self._t_min_d_q(q));                           # momentum of closest point on desired trajectory 
        Sig = torch.sign(torch.dot(p_fin,p.squeeze())); 
        
        
        dq = self._q_t(self._t_min_d_q(q)) - q
        dp = Sig*p_fin - p
        return torch.cat((dq,dp),-1)
    
    def _set_control_parameters(self, a_E,a_M,x_fun,x_num):
        self.a_E = a_E
        self.a_M = a_M
        x0 = x_fun(self.T*0)
        q0 = x0.squeeze()[:2].unsqueeze(0)
        p0 = x0.squeeze()[2:].unsqueeze(0)
        self.E_des = self._energy(q0,p0)
        self.x_t = x_fun
        self.x_num = x_num
        return 0
    
    def _set_damping(self,b):
        self.b = b
        return 0
    
    def _mass_tensor(self,q):
        q2 = q[:,1].unsqueeze(-1)
        M_11 = 2*torch.cos(q2)*l1*l2*m2+l2**2*m2 +l1**2*(m1+m2)
        M_12 = l2*(torch.cos(q2)*l1+l2)*m2
        M_22 = l2**2*m2
        
        return torch.tensor(((M_11,M_12),(M_12,M_22))).to(q)
    
    def _inv_mass_tensor(self,q):
        q2 = q[:,1].unsqueeze(-1)
        M_11 = 2*torch.cos(q2)*l1*l2*m2+l2**2*m2 +l1**2*(m1+m2)
        M_12 = l2*(torch.cos(q2)*l1+l2)*m2
        M_22 = l2**2*m2
        DEN = M_11*M_22-M_12**2
        iM_11 = M_22/DEN #1/l1**2/DEN
        iM_12 = -M_12/DEN
        iM_22 = M_11/DEN #(2*torch.cos(q2)*l1*l2*m2+l2**2*m2 +l1**2*(m1+m2))/(l1**2*l2**2*m2)/DEN
        
        return torch.tensor(((iM_11,iM_12),(iM_12,iM_22))).to(q)


class StabilizedSystemDoublePendulumCosimo(nn.Module):
    # Only changes in _stabilizing_control
    # Elastic Double Pendulum Model
    def __init__(self, V, a_M=torch.tensor(0.0), a_E=torch.tensor(0.0),x_fun = lambda t: torch.tensor((0, 0, 0, 0)).float(), x_num = torch.tensor(((0, 0, 0, 0),(0, 0, 0, 0))).float() ):
        super().__init__()
        self.V, self.T, self.n = V, torch.nn.Parameter(torch.tensor([1.0])), 2
        self._set_control_parameters(a_E, a_M, x_fun, x_num) # Mode control- and energy-injection gains, desired mode 
        self.b = 0 # damping coefficient

    def forward(self, t, x, V_only=False):
        if V_only:
            q = x[...].view(-1, self.n)
            return torch.cat([self._potential_shaping(q), self._energy_shaping(q.requires_grad_(True))],dim=1)
        else:
            # Evaluates the closed-loop vector field
            with torch.set_grad_enabled(True):
                q1, q2, p1, p2 = x[..., 0].view(-1, 1), x[..., 1].view(-1, 1), x[..., 2].view(-1, 1), x[..., 3].view(-1, 1)
                q1 = q1.requires_grad_(True)
                q2 = q2.requires_grad_(True)
                # compute control action
                q = torch.cat([q1, q2], dim=1)
                p = torch.cat([p1, p2], dim=1)
                u = self._energy_shaping(q) + self._stabilizing_control(q,p)
                # compute dynamics
                dxdt = torch.abs(self.T[0]) * self._dynamics(q, p, u)

        return dxdt

    def _dynamics(self, q, p, u):
        u1 = u[:,0].unsqueeze(-1)
        u2 = u[:,1].unsqueeze(-1)
        q1 = q[:,0].unsqueeze(-1)
        q2 = q[:,1].unsqueeze(-1)
        p1 = p[:,0].unsqueeze(-1)
        p2 = p[:,1].unsqueeze(-1)
        


        # NEW controlled elastic pendulum dynamics: (one spring, no damping)
        DET = (l1**2 * l2**2 * m2**2 * (-torch.cos(q2)**2) + l1**2 * l2**2 * m2**2 + l1**2 * l2**2 * m1 * m2)

        dq1dt = (l2**2 * m2 * p1) / DET - (l2 * m2 * p2 * (l1 * torch.cos(q2) + l2)) / DET
        dq2dt = (p2 * (2 * l2 * l1 * m2 * torch.cos(q2) + l1**2 * (m1 + m2) + l2**2 * m2))/ DET - (l2 * m2 * p1 * (l1 * torch.cos(q2) + l2))/ DET

        dp1dt = (-g) * (torch.sin(q1 + q2) * l2 * m2 + torch.sin(q1) * l1 * (m1 + m2)) + u1 - self.b*dq1dt
        dp2dt = -((1/(l1**2 * l2**2 * (m1+torch.sin(q2)**2 * m2)**2)) * ((-p2**2) * torch.sin(q2) * (torch.cos(q2)*l1+l2) * (torch.cos(q2)*l2*m2+l1*(m1+m2))\
        + (1/2)* p1 * p2 * torch.sin(q2) * l2 * (4*torch.cos(q2)*l2*m2+l1*(2*m1+(3+torch.cos(2*q2))*m2))\
        + l2**2 * ((-torch.cos(q2)) * p1**2 * torch.sin(q2) * m2 + l1**2 * (m1+torch.sin(q2)**2 * m2)**2 *(g*torch.sin(q1+q2)*l2*m2-k1*(torch.pi-2*q2)))))\
        + u2 - self.b*dq2dt
        

        return torch.cat([dq1dt, dq2dt, dp1dt, dp2dt], dim=1)

    def _energy_shaping(self, q):
        # energy shaping control action
        dVdx = grad(self.V(q).sum(), q, create_graph=True)[0]
        return -dVdx

    def _potential_shaping(self, q):
        # energy shaping control action
        V = self.V(q)
        return V
    
    def _autonomous_potential(self,q):
        q1 = q[:,0].unsqueeze(-1)
        q2 = q[:,1].unsqueeze(-1)
        return -g*(torch.cos(q1 + q2)*l2*m2 + torch.cos(q1)*l1*(m1 + m2)) + k1*(1/2*torch.pi - q2)**2
        
    def _autonomous_kinetic(self,q, p):
        return 1/2*torch.inner(p.squeeze(),self._inv_mass_tensor(q)@p.squeeze())

    def _autonomous_energy(self, q, p):
        # Hamiltonian (total energy) of the UNCONTROLLED system
        q1 = q[:,0].unsqueeze(-1)
        q2 = q[:,1].unsqueeze(-1)
        Pot = -g*(torch.cos(q1 + q2)*l2*m2 + torch.cos(q1)*l1*(m1 + m2)) + k1*(1/2*torch.pi - q2)**2
        Kin = 1/2*torch.inner(p.squeeze(),self._inv_mass_tensor(q)@p.squeeze())
        return Pot+Kin

    def _energy(self, q, p):
        # Hamiltonian (total energy) of the CONTROLLED system
        return self._autonomous_energy(q,p)+self._potential_shaping(q).sum()
    
    def _stabilizing_control(self,q,p):
        # mode and energy stabilizing control action
        #self.q_t,self.p_t,self.E_Des,self.T
        
        pnorm = torch.sqrt(torch.dot(p.squeeze(),self._inv_mass_tensor(q)@p.squeeze()))
        if pnorm > 0:
            p_hat = p/pnorm;
        else:
            p_hat = p;
        
        t_min = self._t_min_d_q(q);
        q_fin = (self._q_t(t_min)).unsqueeze(0);
        p_fin = self._p_t(t_min);                                                # momentum of closest point on desired trajectory 
        Sig = torch.sign(torch.dot(p_fin,p.squeeze()));                          # Sign depends on current system momentum
        p_des_raw = Sig*p_fin;                                                   # desired momentum component
        
        p_des =(p_des_raw-p);                                                    #  torch.matmul(self._mass_tensor(q)*self._inv_mass_tensor(q_fin),p_des_raw) - p; # 
                                                                                 # UNPROJECTED VESION:  p_des = (p_des_raw-p) 
                                                                                 # COSIMO'S ACTUAL VERSION: p_des = M(q)(dqdt_des - dqdt) = M(q)*M(q_des)^(-1)*p_des - p; 
                                                                                 #     or as code: torch.matmul(self._mass_tensor(q)*self._inv_mass_tensor(q_fin),p_des_raw) - p 
                                                                                 #     This did not work.
        F_M = self.a_M*p_des;                                                    # Mode stabilizing control
        
        Diff_E = self.E_des - self._energy(q,p)
        F_E = self.a_E*Diff_E*p_hat;#*pnorm/(pnorm**2 + 1e-9);                   # Energy injection control
        F = F_E + F_M;
        
        return F

    def _q_t(self, t):
        return self.x_t(t).squeeze()[:2]

    def _p_t(self, t):
        return self.x_t(t).squeeze()[2:]

    def _t_min_d_q(self, qc):
        #time of minimum approach of trajectory self._q_(t) from qc
        len_xnum = torch.tensor(self.x_num).size()[0]
        i = torch.argmin(torch.norm(torch.tensor(self.x_num[:np.int(np.floor(len_xnum/2)),:2]).to(qc)-qc,dim=1))
        # i = torch.argmin(torch.norm(torch.tensor(self.x_num[:np.int(len_xnum/2),:2]).to(qc)-qc,dim=1))
        T_rough = self.T*i/torch.tensor(self.x_num[...,:2]).size()[0]
        Err_min = 1e-4
        q = self._q_t(T_rough);
        dq = numJ(self._q_t,T_rough,Err_min).squeeze();
        ddq = numJ(lambda x: numJ(self._q_t,x,Err_min).squeeze(), T_rough, Err_min).squeeze();
        p = torch.cat((2*torch.mul(ddq,ddq).unsqueeze(-1),
                       3*torch.mul(ddq,dq).unsqueeze(-1),
                       (2*torch.mul(ddq,(q-qc.squeeze()))+torch.mul(dq,dq)).unsqueeze(-1),
                       torch.mul(dq,(q-qc.squeeze())).unsqueeze(-1)),-1)
        p = p[0,:]+p[1,:];
        r = cuberoot(p) # used to return all roots in matlab
        dm = torch.norm(q-qc,dim=1);
        dt = torch.zeros(1).to(T_rough)
        #for i in range(1): # used to check all roots, best end-point of interval if none are better
        qm = self._q_t(T_rough+r);
        dm_new = torch.norm(qm-qc,dim=1);
        if dm_new[0] < dm[0]:
            dm = dm_new;
            dt = r;

        T_fine = T_rough+dt;

        return T_fine
    
    
    def _min_d(self,qc):
        return torch.norm(qc - self._q_t(self._t_min_d_q(qc)))
    
    def _q_des(self,q):
        # desired position at current point
        return self._q_t(self._t_min_d_q(q))
    
    def _p_des_raw(self,q,p):
        # desired final momentum at current point
        
        p_fin = self._p_t(self._t_min_d_q(q));                           # momentum of closest point on desired trajectory 
        Sig = torch.sign(torch.dot(p_fin,p.squeeze()));                                  # Sign depends on current system momentum
        return Sig*p_fin; 
    
    def _phase_dist(self,q,p):
        t = self._t_min_d_q(q)
        p_fin = self._p_t(self._t_min_d_q(q));                           # momentum of closest point on desired trajectory 
        Sig = torch.sign(torch.dot(p_fin,p.squeeze())); 
        
        
        dq = self._q_t(self._t_min_d_q(q)) - q
        dp = Sig*p_fin - p
        return torch.cat((dq,dp),-1)
        
    
    def _set_control_parameters(self, a_E,a_M,x_fun,x_num):
        self.a_E = a_E
        self.a_M = a_M
        x0 = x_fun(self.T*0)
        q0 = x0.squeeze()[:2].unsqueeze(0)
        p0 = x0.squeeze()[2:].unsqueeze(0)
        self.E_des = self._energy(q0,p0)
        self.x_t = x_fun
        self.x_num = x_num
        return 0
    
    def _set_damping(self,b):
        self.b = b
        return 0
    
    def _mass_tensor(self,q):
        q2 = q[:,1].unsqueeze(-1)
        M_11 = 2*torch.cos(q2)*l1*l2*m2+l2**2*m2 +l1**2*(m1+m2)
        M_12 = l2*(torch.cos(q2)*l1+l2)*m2
        M_22 = l2**2*m2
        
        return torch.tensor(((M_11,M_12),(M_12,M_22))).to(q)
    
    def _inv_mass_tensor(self,q):
        q2 = q[:,1].unsqueeze(-1)
        M_11 = 2*torch.cos(q2)*l1*l2*m2+l2**2*m2 +l1**2*(m1+m2)
        M_12 = l2*(torch.cos(q2)*l1+l2)*m2
        M_22 = l2**2*m2
        DEN = M_11*M_22-M_12**2
        iM_11 = M_22/DEN #1/l1**2/DEN
        iM_12 = -M_12/DEN
        iM_22 = M_11/DEN #(2*torch.cos(q2)*l1*l2*m2+l2**2*m2 +l1**2*(m1+m2))/(l1**2*l2**2*m2)/DEN
        
        return torch.tensor(((iM_11,iM_12),(iM_12,iM_22))).to(q)
