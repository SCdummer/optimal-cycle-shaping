These files include the Generators and Eigenmodes for a Double Pendulum in gravity.

The double pendulum always has: point-emasses of mass 0.4kg, mass-less links of length 1m

There is a rotational spring between the first and the second link: this adds an extra potential term.
The cases k = 0, k = 1 and k = 10 were simulated. 

Generators for k = n are found in: 

Generators_DP_k_n 
- 2 by 3 cell
- cell {n,1} contains a generator of the n-th eigenmanifold with column 1: q1(t=0), column 2: q2(t=0), column 3: Potental energy(t=0)
- cell {n,2} contains the period of the corresponding mode of the n-th eigenmanifold
- cell {n,3} contains yet again, the energy of the corresponding mode of the n-th eigenmanifold

Trajectories for k = n are found in:

Trajectories_DP_k_n
- 2 by 4 cell
- cell {n,1}{k} contains all k-th trajectory of n-th eigenmanifold: column 1: q1, column 2: q2, column 3: dq1/dt, column 4: dq2/dt
- cell {n,2} is irrelevant (a type of modal coordinate)col
- cell {n,3}{k} contains time t along k-th trajectory of n-th eigenmanifold
- cell {n,4} is empty

Potentials V(q) for k = n are found in:
DP_k_n/V.m
