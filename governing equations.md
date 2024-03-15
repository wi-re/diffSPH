# Algorithm (no free surface)

0. Neighborsearch with velocity verlet
1. Compute $L_i$
2. Compute $\nabla^L\rho_i$
3. Compute $\delta$ diffusion for density
4. Compute $\alpha$ diffusion for velocity (inviscid)
5. Compute Momentum Eqn. $\nabla \cdot \mathbf{u}$
6. Compute EOS $p(\rho)$
7. Compute pressure accel $\frac{1}{\rho}\nabla p$
8. Compute divergence term $\nabla \cdot \mathbf{u}$
9. Compute SPS term
10. update $\frac{du}{dt}$ and $\frac{d\rho}{dt}$

Repeat 1-10 4 times, compute RK update, compute Shift term

# Neighborsearch

Assuming an ideal oracle for the neighborsearch that already provides each particle cell association.

Each cell is of size:

$$\Delta x_\text{cell} = \operatorname{max} h_i \frac{H}{h} \sigma_\text{verlet}$$

Assuming constant support radii $h_i = 2 \Delta x$ we get

$$\Delta x_\text{cell} = 2 \frac{H}{h} \Delta x \sigma_\text{verlet}$$

For Wendland 2 $\frac{H}{h}\approx 1.897367$. Each cell now has an area of 

$$A_\text{cell} = 4 \left(\frac{H}{h}\right)^2 \Delta x^2 \sigma_\text{verlet}^2$$

With each particle having an area of $\Delta x^2$. Accordingly, the expected number of particles per cell is 

$$N_\text{cell} = 4 \left(\frac{H}{h}\right)^2 \sigma_\text{verlet}^2$$

Whereas the expected number of neighbors for a particle is based on $\pi (2\Delta x \frac{H}{h})^2$:
$$N_\text{ptcl} = \frac{4 \pi \Delta x^2}{\Delta x^2}\left(\frac{H}{h}\right)^2 = 4\pi \left(\frac{H}{h}\right)^2$$.

For Wendland 2 we thus get $N_\text{ptcl} \approx 45.22$. The ratio of $9 N_\text{cell}$ ($\pm1$ cell for the search) to $N_\text{ptcl}$ is 

$$\xi = \frac{9 \cdot 4 \left(\frac{H}{h}\right)^2 \sigma_\text{verlet}^2}{4\pi \left(\frac{H}{h}\right)^2} = \frac{9}{\pi}\sigma_\text{verlet}^2$$

For convenience we use $\sigma_\text{verlet} = \sqrt{2}$, thus $\xi = \frac{18}{\pi}\approx 5.79$, i.e., the percentage of actual neighbors is $17.4\%$. For heuristic arguments let $\xi = 6$. 

We now introduce the concept of a cell graph and a particle graph for brevity. A cell graph is a graph where all particles are vertices and the connectivity is based on the particles per cell, whereas the particle graph only connects particles within the support radii, i.e., we consider $N_\text{ptcl}$ and $\xi N_\text{ptcl}$.

Accordingly, any operation performed per particle results in $n_{\text{ptcl}}$ actual operations, an operation performed per neighbor results in $n_\text{neigh} = n_\text{ptcl} N_H = n_\text{ptcl} N_\text{ptcl}$ operations and an operation performed per potential neighbor results in $n_{cell} = n_\text{ptcl}\xi N_\text{ptcl}$ operations. For further convenience let:

(a) $n_\text{neigh} = N_H n_\text{ptcl}$
(b) $n_\text{cell} = \xi N_H n_\text{ptcl}$


In our neighbor search we compute the distance for all $n_\text{cell}$ particles, compare them to $H$, sum up the number of neighbors and then recompute this for the actual neighbor list creation, i.e., we get:

1. Compute Distance: $d\cdot n_\text{cell}$ multiplications, $(d-1)\cdot n_\text{cell}$ additions, $n_\text{cell}$ square roots 
2. Check distance: $n_\text{cell}$ comparisons
3. Summation of neighbor counters $n_\text{neigh}$
4. Cummulative Summation $n_\text{ptcl}$
1. Compute Distance: $d\cdot n_\text{cell}$ multiplications, $d-1\cdot n_\text{cell}$ additions, $n_\text{cell}$ square roots 
2. Check distance: $n_\text{cell}$ comparisons

Overall: 
- $2\cdot d\cdot n_\text{cell}$ multiplication
- $2\cdot (d-1)\cdot n_\text{cell} + n_\text{neigh} +  n_\text{ptcl}$ additions
- $2\cdot n_\text{cell}$ square roots
- $2\cdot n_\text{neigh}$ comparisons

Assuming the cost of all operations is identical we thus get
$$\operatorname{FLOP}_\text{NN} = (4d \xi N_H+ 2N_H + 1) \cdot n_\text{ptcl}$$


# Neighbor computations

We now compute the relavant particle quantities:
```py
xij = pos_x[i] - pos_y[j]
xij = torch.stack([xij[:,i] if not periodic_i else mod(xij[:,i], minD[i], maxD[i]) for i, periodic_i in enumerate(periodicity)], dim = -1)
rij = torch.sqrt((xij**2).sum(-1))
xij = xij / (rij + 1e-7).view(-1,1)
rij = rij / hij
```

As we assume the floating point cost of everything to be identical, from now on, 

1. $d\cdot n_\text{neigh}$
2. $11d\cdot n_\text{neigh}$
3. $2d\cdot n_\text{neigh}$
4. $dn_\text{neigh}$
5. $n_\text{neigh}$

In total: $(15d+1)N_H n_{\text{ptcl}}$

```
Wij = kernel.kernel(rij, hij, dim)
gradWij = kernel.kernelGradient(rij, xij, hij, dim) 
```

Assuming we use a Wendland 2 kernel the actual kernel computations require (assuming ^3 and ^4 operations require 2 FLOP) with precomputed distances:

1. Kernel: 11 FLOP
2. Gradient: 20 FLOP

So in total the neighbor computations require

$$\operatorname{FLOP}_{NC} = (32 + 15 d) N_H n_{\text{ptcl}}$$

For a neural network we can drop the $32$ operations for the kernel functions.

In total the neighbor search then requires 

$$\operatorname{FLOP}_\text{Nsearch} = ((34 + 15 d + 4d \xi)N_H + 1) \cdot n_\text{ptcl}$$

with $\xi = \frac{9}{\pi}\sigma^2_\text{verlet}$ and $N_H = 4\pi \left(\frac{H}{h}\right)^2$ we get

$$ ((34 + 15 d + 4d \xi)N_H + 1) = (34 + 15 d + 4d \frac{9}{\pi}\sigma^2_\text{verlet})4\pi \left(\frac{H}{h}\right)^2 + 1$$

Assuming $d=2$ (from now on as this only concerns 2d operations):

$\left(256\pi + 72 \sigma^2_\text{verlet} \right) \left(\frac{H}{h}\right)^2 + 1$

Using $\sigma = \sqrt{2}$ and $H/h = 1.897367$ we then get

$\left(256\pi + 144 \right) 1.897367^2 + 1\approx 3415$

Floating operations per particle in SPH and $1967$ for a neural network.

# $L_i$:

Distance computation: 4 $n_\text{neigh}$
Gradient Operation (difference): 2 + 1, einsum: 2^2, sum 4 = 11 $n_\text{neigh}$
pinv2x2:

theta: 13 + atan2 + cos + sin + 1
S1: 7
S2: 15
phi: 13 + atan2 + cos + sin
s11: 10
s22: 11
V: 5
sigma: 6
matmul: 16
total: 96 + 2 atan2 + 2 cos + 2 sin, assuming trig = 2 ops: 108

in total: $123 n_\text{neigh}$

2. Compute $\nabla^L\rho_i$

8 (normalized Gradients)
20 (matrix div)
in total: $28 n_\text{neigh}$

3. Compute $\delta$ diffusion for density

13 (precompute) n_neigh
7 (vector div) n_neigh
4 (post process) n_ptcl

in total: $20 n_\text{neigh} + 4 n_\text{ptcl}$


4. Compute $\alpha$ diffusion for velocity (inviscid)

14 n_neigh
5 n_ptcl

5. Compute Momentum Eqn. $\nabla \cdot \mathbf{u}$
7 n_neigh
2 nptcl

6. Compute EOS $p(\rho)$
3 n_ptcl

7. Compute pressure accel $\frac{1}{\rho}\nabla p$
3 n_ptcl
9 (scalar grad) n_neigh

8. Compute divergence term $\nabla \cdot \mathbf{u}$
7 n_neigh

9. Compute SPS term
11 (vector grad) n_neigh
26 n_ptcl matrix computation
13 n_neigh summation

total: 24 n_neigh + 26 n_ptcl

10. update $\frac{du}{dt}$ and $\frac{d\rho}{dt}$
3 n_ptcl

In summary we get:

Function | FLOP
---|---
Nearest Neighbor | $(4d \xi N_H+ 2N_H + 1) \cdot n_\text{ptcl}$
Neighbor Computation | $(32 + 15 d) N_H n_{\text{ptcl}}$
$L_i$ | $123 n_\text{neigh}$
$\nabla^L\rho_i$ | $28 n_\text{neigh}$
$\delta$ | $20 n_\text{neigh} + 4 n_\text{ptcl}$
$\alpha$ | $14 n_\text{neigh} + 5 n_\text{ptcl}$
Momentum | $7 n_\text{neigh} + 2 n_\text{ptcl}$
EOS | $3 n_\text{ptcl}$
$\nabla p$ | $9 n_\text{neigh} + 3 n_\text{ptcl}$
$\nabla \dot u$ | $7 n_\text{neigh}$
SPS | $24 n_\text{neigh} + 26 n_\text{ptcl}$
Update | $3 n_\text{ptcl}$

THus for the SPH operations we get:
$232 n_\text{neigh} + 46 n_\text{ptcl}$ and $64 n_\text{neigh}$ from the neighbor computations and $(8\xi + 2)n_\text{neigh} + n_{ptcl}$ from the nearest neighbor. This in total gives

$$296 N_H + 46$$ without nearest neighbor. Assuming $N_H = 45.22$ we get $13431$ FLOP per particle. And $$8\xi N_H 298 N_H + 47$$ with nearest neighbor. Assuming $N_H = 45.22$ and $\xi = 6$ we thus get $15693$ FLOP per particle

Using RK4 and assuming 1 nearest neigbhor per overall step we need $55986$ FLOP per particle. We also need to add 

We also get a receptive field of 8 for each substep + 1 for the shifting

We also have 8 gradient computations assumed free as they are stored, if we computed them (distance and sqrt included) we need an additional $8N_h n_\text{ptcl}\cdot 29$ FLOP or 3255 FLOP per substep or 13023 in total extra. This gives us 69009 FLOP per particle. Including an additional overhead of 15% to account for more expensive operations (such as trig functions, particle shifting, neighbor list data structure generation writing to arrays etc) we get approximately 80K FLOP per particle.

Meanwhile a neural network needs ```py
    FloatingOperations = edges * (parameterDict['Basis Terms']**parameterDict['Dimensionality'] + 2 * parameterDict['Basis Terms']**parameterDict['Dimensionality'] 
                                  * parameterDict['Input Features'] * parameterDict['Output Features'] + parameterDict['Input Features']*parameterDict['Output Features'] - parameterDict['Output Features'])``` per CConv.
So for 4 basis terms in 2D we get 
$$ N_H n_\text{ptcl} \left(16 + 31 * io - o\right) = 16 N_H n_\text{ptcl} + N_H n_\text{ptcl} \left(31 * io - o\right)$$

For a classic CConv with (ignoring linear layers for brevity) we get $32->32$ features, $96->64$, $64->64$ and $64->2$ gives 

31 * 32 * 32 - 32 =   31712
31 * 96 * 64 - 64 =  190400
31 * 64 * 64 - 64 =  126912
31 * 64 * 2 - 2   =  3966
= 352990 + 16
(this is too small ,should be 33)
which are approximately 15 962 931 FLOP per particle. This is an increase of approximately $200\times$ with a reduction in receptive field from $8$ to $4$.


With 4 instead of 32 we would get:
33 * 4 * 4 - 4  = 524
33 * 12 * 8 - 8 = 3160
33 * 8 * 8 - 8  = 2104
33 * 8 * 2 - 2  = 526
= 6330
which is approximately 286 242 FLOP per particle. This is an increase of only $3.5\times$