#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main routine to simulate motion of Argon atoms under constant pressure in four dimensions.
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import time
import matplotlib



#LJ constants
e=0.01 #eV Ar
sigma = 3.4 #A Ar
m=40/0.00964855 #mass of particles in eV, mass in amu *0.00964855
cutoff = 15 #cutoff radius for cells interacting (in A) must be at most neighbour_range*cell_l
dim = 3 #dimensions of space of the simulation
side=5 #number of particles in side of initial lattice
N=side**dim #number of particles in cell
spacing = 4 #initial spacing of the lattice
l = spacing*(side+1) #length of initial lattice side
cell_l = l # length of cell boundary
N=side**dim
dt = 0.5
target_T= 10 #in K
period = 150 #as estimated for Ar
runtime = 100 # time simulation is ran for
steps = int(math.floor(runtime/dt)) #number of steps to integrate for, 10*period time
#steps = 1000
V0 = cell_l**dim
cell_l0 = V0**(1/dim)
neighbour_range = 1 #neighbouring cells up to neighbours away included in potential
print("want negative: ", cutoff - neighbour_range*cell_l0)
W = 10**6
P_t = 10**10/1.6022e11 #convert from Pa
#P_t =0
dr = 0.05
rad_array = np.arange(11+dr, step=dr)
mean_rad_array = (rad_array + 0.5*dr)[:-1]

def LJ(Ri, Rj):
    """Compute Lennard-Jones potential value for particles at positions Ri, Rj."""

    Rij = Rj-Ri
    rij = np.sqrt(Rij.dot(Rij))

    return 4*e*((sigma/rij)**12-(sigma/rij)**6)

def LJ_der(Ri, Rj):
    """Compute derivative vector dV/dRi of LJ potential given particle
    positions Ri, Rj."""

    Rij = Rj-Ri
    rij = np.sqrt(Rij.dot(Rij))
    der = 4*e*(6*sigma**6/rij**8-12*sigma**12/rij**14)*Rij

    return der


def lattice(a=1, side=10, dim=2):
    """Generates initial positions R of lattice of particles with 10^2 particles
    and lattice spacing a. Positions vary from 0 to a*(side-1)."""

    R = []
    if dim == 2:
        for i in range(side):
            for j in range(side):
                R.append([(i+1)*a,(j+1)*a])

    if dim ==3:
        for k in range(side):
            for i in range(side):
                for j in range(side):
                    R.append([(i+1)*a,(j+1)*a, (k+1)*a])

    return np.array(R)

def neighbour_list(neighbour_range=1, dim=2):
    """Find list of vectors b describing displacement to neighbouring cells."""
    list = []

    if dim ==2:
        for i in range(-neighbour_range,neighbour_range+1):
            if not i==0:
                list.append([i,i])
            for j in range(-neighbour_range,neighbour_range+1):
                if not i==j:
                    list.append([i,j])

    if dim ==3:
        for i in range(-neighbour_range,neighbour_range+1):
            if not i==0:
                list.append([i,i,i])
            for j in range(-neighbour_range,neighbour_range+1):
                for k in range(-neighbour_range,neighbour_range+1):
                    if not (i==j and i==k):
                        list.append([i,j,k])

    return np.array(list)

def Boltzmann(N, dim, target_T):
    """Boltzmann dist for initial velocities for N particles in N dimensions
    for temperature T in K."""

    kB= 8.617*10**(-5) #Boltzmann constant
    kT = kB*target_T
    a = np.sqrt(kT/m)

    #normal dist for velocity components
    Iv = np.zeros((N, dim))
    for i in range(N):
        for j in range(dim):
            Iv[i,j]=random.gauss(0,a)

    return Iv

def fit_in_cell_slow(cell_l, s):
    """Take a set of s and translate all back to unit cell using periodic 
    boundary conditions. """
    N = len(s)
    dim = len(s[0,:])
    
    for i in range(N):
        for j in range(dim):                
            if s[i,j] > cell_l:
                print("Moved atom!")
                if abs(s[i,j] - cell_l) >= cell_l:
                    print("Molecule 2 cells away or more!")
                    
                #s[i,j] -= cell_l
                s[i,j] = (s[i,j]%cell_l)
                
            if s[i,j] < 0:
                print("Moved atom!")
                if abs(s[i,j] + cell_l) >= cell_l:
                    print("Molecule 2 cells away or more!")
                    
                #s[i,j] += cell_l
                s[i,j] = (s[i,j]%cell_l)
    
    return s

def fit_in_cell_old(dim, N, cell_l, s):
    """Take a set of s and translate all back to unit cell using periodic 
    boundary conditions. """
    
    s_copy =s.copy()
    s_minus_cell = s_copy -cell_l0
    translate_back = s_minus_cell > 0
    translate_forward = s < 0
    
    s_copy[translate_back] -= cell_l
    s_copy[translate_forward] += cell_l
    
    if translate_back.any() or translate_forward.any():
        print("Moved atom!")
        
    return s_copy


def fit_in_cell(dim, N, cell_l, s):
    """Take a set of s and translate all back to unit cell using periodic 
    boundary conditions. """
    
    s_copy =s.copy()
    s_copy %= cell_l
        
    return s_copy

def get_pair_correlation_fn(dim, N, all_r, rad_array, cell_l):
    """Calculate the series of the pair correlation function from the physical
    postitions r, dividing into num bins."""
    
    num = len(rad_array)-1 #number of radius intervals
    ref_i = int(N/2) #index of reference particle
    n= N/(cell_l**dim)
    
    
    displacements = all_r - all_r[ref_i,:] #distances to reference particle
    distances = np.linalg.norm(displacements, axis=1) #magnitudes of each row - particle

    distances = np.delete(distances, ref_i)

    num_array = np.zeros(num) #count how many particles are in between the radii
    

    for i in range(len(distances)):
        distance_to_i = distances[i]
        for j in range(num):
            if distance_to_i >= rad_array[j] and distance_to_i < rad_array[j+1]:
                num_array[j] += 1

    dr = rad_array[1]- rad_array[0]
    g = num_array/(dr*n)
    
    
    return g

        
        
def a_lam_PBC(dim, N, s, lam, V0, P_t, neighbour_b, cell_l0, cutoff, rad_array):
    """Compute acceleration due to Lennard-Jones potential given positions."""
    a_lam = 0
    p_pot = 0
    R_cutoff = np.zeros(dim)
    R_cutoff[0] = cutoff
    V_cutoff = LJ(np.zeros(dim), R_cutoff)
    V_atoms =  -V_cutoff
    
    N_cells = len(neighbour_b) #number of neighbouring cells
    N_all = N*(N_cells+1) #number of particles in local + neighbouring cells
    acc_list = np.zeros((N,N*(N_cells+1),dim))
    
    s_in_cell = fit_in_cell(dim, N, cell_l0, s)

    neighbour_s = neighbour_b*cell_l0 #displacement of neighbour cells
    all_s = np.zeros((N_all,dim)) #array of positions of particles in all cells
    all_s[:N,:] = s_in_cell #first N rows is local cell
    all_r = np.zeros((N_all,dim)) #array of positions of particles in all cells
    all_r[:N,:] = s #first N rows is local cell

    #add positions of particles in neighbouring cells
    for i in range(1,N_cells+1):
        all_s[N*i:N*(i+1),:] = s_in_cell+np.full((N,dim),neighbour_s[i-1,:])
        all_r[N*i:N*(i+1),:] = s+np.full((N,dim),neighbour_s[i-1,:])
    
    all_r*=lam
    g = get_pair_correlation_fn(dim, N, all_r, rad_array, cell_l0*lam)
    
    #add in contribution of the potential derivative for j=/=i
    for i in range(N):
        ri = lam*all_s[i,:]
        for j in range(i+1,N*(N_cells+1)):
                rj = lam*all_s[j,:]
                if (ri-rj).dot(ri-rj) <= cutoff**2:
                    Fi = LJ_der(ri, rj)
                    acc_list[i,j,:] = lam*Fi
                    a_lam_term = Fi.dot(all_s[i,:]) - Fi.dot(all_s[j,:])
                    
                    if j<N: # atoms in the base cell
                        acc_list[j,i,:] = -lam*Fi
                        V_atoms += LJ(ri, rj)
                        p_pot += a_lam_term #only add base cell terms to pressure
                    else: #j in image cell
                        V_atoms += 0.5*LJ(ri, rj)
                        p_pot += 0.5*a_lam_term
    
    p_pot *= lam**(1-dim)/(V0*dim)
    a_lam = (p_pot-P_t)*dim*V0*lam**(dim-1)/W
    a_s = np.sum(acc_list,axis=1)/m
    V_cell = P_t*V0*lam**dim
    
    return a_s, a_lam, p_pot, V_atoms, V_cell, g

def integrate(dim, N, Is, Ivs, Ilam, Ivlam, timesteps, V0, P_t, dt,neighbour_b, cell_l0,  cutoff, rad_array):
    """Perform leapfrog integration with steo dt for timesteps number of steps."""
    s = Is
    vs = Ivs
    lam = Ilam
    vlam = Ivlam

    energies = [] # energy timeseries
    s_series=[] # position series
    lam_series = []
    pressures = [] #diagonal elements of the stess tensor series\    s_series=[] # position series
    mss_series = [] #mean square s 
    g_series = [] #pair correlation function
    
    #initial Verlet timestep to get v(t-dt/2)
    print("pre step")
    vs, vlam = timestep_V(dim, N, Is, Ivs, Ilam, Ivlam, V0, P_t, -0.5*dt, neighbour_b, cell_l0,  cutoff,rad_array)
    s_series.append(s)

    #progress solution by "steps" timesteps
    print("start integration")
    for i in range(timesteps):
        s, vs,lam, vlam, p, V_atoms, V_cell, T_atoms, T_cell, mss, g = timestep_VV(dim, N, s, vs, lam, vlam, V0, P_t, dt, neighbour_b, cell_l0,  cutoff, rad_array)
        energies.append([V_atoms+V_cell+T_atoms+T_cell,V_atoms, V_cell, T_atoms, T_cell])
        s_series.append(s)
        lam_series.append(lam)
        pressures.append(p)
        mss_series.append(mss)
        g_series.append(g)

        if i%100 == 0:
            print("step ", i)


    return s, vs, lam, vlam, s_series, lam_series, np.array(energies), np.array(pressures), np.array(mss_series), np.array(g_series)



def timestep_VV(dim, N, s, vs, lam, vlam, V0, P_t, dt, neighbour_b, cell_l0,  cutoff, rad_array):
    """Perform timestep using velocity Verlet algorithm.
    INPUT:
    R - positions R(t)
    v - velocities R(t-dt/2)
    dt - timestep"""

    a_s, a_lam, p_pot, V_atoms, V_cell, g = a_lam_PBC(dim, N, s, lam, V0, P_t, neighbour_b, cell_l0,  cutoff, rad_array)

    vs_half_step = vs + a_s*dt #v(t+dt/2)
    vlam_half_step = vlam + a_lam*dt
    new_s = s + vs_half_step*dt #R(t+dt)
    new_lam = lam + vlam_half_step*dt

    #compute velocities at t
    new_vs = 0.5*(vs+vs_half_step)
    new_vlam = 0.5*(vlam+vlam_half_step)
    
    T_atoms = 0.5*m*np.sum(new_vs**2)
    T_cell = 0.5*W*new_vlam**2
    
    mss = np.sum((new_lam*new_s-Is)**2)/N # mean square s
    

    return new_s, vs_half_step,new_lam, vlam_half_step, p_pot, V_atoms, V_cell, T_atoms, T_cell, mss, g

def timestep_V(dim, N, s, vs, lam, vlam, V0, P_t, dt, neighbour_b, cell_l0,  cutoff, rad_array):
    """Perform Verlet timestep.
    R - positions R(t)
    v - velocities v(t)
    dt - timestep"""


    a_s1, a_lam1, p_pot, V_atoms, V_cell, g= a_lam_PBC(dim, N, s, lam, V0, P_t, neighbour_b, cell_l0,  cutoff, rad_array)
    new_s = s + vs*dt + 0.5*a_s1*dt**2 #R(t+dt)
    new_lam = lam + vlam*dt + 0.5*a_lam1*dt**2
    a_s2, a_lam2,p_pot, V_atoms, V_cell, g = a_lam_PBC(dim, N, new_s, new_lam, V0, P_t, neighbour_b, cell_l0,  cutoff, rad_array)
    new_vs = vs + 0.5*(a_s1+a_s2)*dt
    new_vlam = vlam + 0.5*(a_lam1+a_lam2)*dt

    return new_vs, new_vlam

#%%
np.random.seed(2)
random.seed(2)
#Routine 1:
#lattice IC with small random deviations
#Is = lattice(a=spacing,side=side, dim=dim)+(np.random.random((N,dim))-0.5)*spacing*0.05

#Routine 2:
lamfin = 0.906
cell_l_scaled=cell_l*lamfin*0.68
Is = fit_in_cell(dim, N, cell_l_scaled, lamfin*np.loadtxt("s_dt0.500_W10E6.csv", delimiter=','))


fig, ax = plt.subplots()
for i in range(N):
    plt.plot(Is[i, 0], Is[i, 1],'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Initial positions')
plt.show()

#Routine 1:
#Ivs = Boltzmann(N, dim, target_T)

#Routine 2:
Ivs = np.loadtxt("vs_dt0.500_W10E6.csv", delimiter=',')

Ilam = 1.0
Ivlam = 0
neighbour_b = neighbour_list(neighbour_range,dim)
# %%
print("Running for ", steps, " steps...")
tic = time.time()
s, vs, lam, vlam, s_series, lam_series, energies, pressures, mss_series, g_series = integrate(dim, N, Is, Ivs, Ilam, Ivlam, steps, V0, P_t, dt, neighbour_b, cell_l_scaled,  cutoff, rad_array)
toc = time.time()
print("Integrated %i steps in %0.4f seconds" %(steps,(toc - tic)))

# %%

save = True
Wpower = np.log10(float(W))

timescale_fs = dt*np.arange(steps)
timescale_ps = dt*np.arange(steps)/10**3


#plot initial positions
fig, ax = plt.subplots()
for i in range(N):
    plt.plot(Is[i, 0], Is[i, 1],'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Initial positions')
plt.show()

#plot final positions
fig, ax = plt.subplots()
for i in range(N):
    plt.plot(s[i, 0], s[i, 1],'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Final positions')
plt.show()


fig, ax = plt.subplots()
plt.plot(timescale_ps, lam_series, color ='orange')
plt.ylabel(r"$\lambda$")
plt.xlabel("Time [ps]")
if save: 
    plt.savefig("lam_series_PBC_dt%0.3f_W10E%i.svg" %(dt,Wpower))
plt.show()


T = energies[:,3]
Temperatures = (2*T)/(dim*(N-1)*8.617*10**(-5))


fig, ax = plt.subplots()
plt.plot(timescale_ps, energies[:,0]/10**3, label="Enthalpy")
plt.plot(timescale_ps, energies[:,3]/10**3, label="Kinetic energy of atoms")
plt.plot(timescale_ps, energies[:,1]/10**3, label="Potential energy of atoms")
plt.plot(timescale_ps, energies[:,4]/10**3, label="Kinetic energy of cell")
plt.plot(timescale_ps, energies[:,2]/10**3, label="PV term")
plt.legend()
plt.ylabel("Energy [keV]")
plt.xlabel("Time [ps]")
if save:
    plt.savefig("energy_series_PBC_dt%0.3f_W10E%i.svg" %(dt,Wpower))
plt.show()

fig, ax = plt.subplots()
plt.plot(timescale_ps, energies[:,0]/10**3)
plt.ylabel("Enthalpy [keV]")
plt.xlabel("Time [ps]")
if save:
    plt.savefig("total_energy_series_PBC_dt%0.3f_W10E%i.svg" %(dt,Wpower))
plt.show()


fig, ax = plt.subplots()
plt.plot(dt*np.arange(steps), Temperatures)
plt.ylabel("Temp [K]")
plt.xlabel("time [fs]")
if save:
    plt.savefig("temperature_series_PBC_dt%0.3f_W10E%i.svg" %(dt,Wpower))
plt.show()


fig, ax = plt.subplots()
plt.plot(timescale_ps, pressures*1.6022e11/10**9, label="Internal pressure")
plt.plot(timescale_ps, np.full(steps, P_t*1.6022e11)/10**9, '--', label="Target pressure")
plt.ylabel("Pressure [GPa]")
plt.legend()
plt.xlabel("Time [ps]")
if save:
    plt.savefig("pressure_series_PBC_dt%0.3f_W10E%i.svg" %(dt,Wpower))
plt.show()


lam_KE_av = np.mean(energies[:,4])

fig, ax = plt.subplots()
plt.plot(timescale_ps, energies[:,4], label="Kinetic energy of cell")
plt.plot(timescale_ps, np.full(steps, lam_KE_av), '--', label='Average kinetic energy of cell')
plt.plot(timescale_ps, energies[:,3]/(dim*N-dim), label="Kinetic energy of atoms/dof")
plt.legend()
plt.ylabel("Energy [eV]")
plt.xlabel("Time [ps]")
if save:
    plt.savefig("energy2_PBC_dt%0.3f_W10E%i.svg" %(dt,Wpower))
plt.show()

fig, ax = plt.subplots()
plt.plot(timescale_ps,mss_series)
plt.ylabel(r"Mean square displacement [$\AA^{2}$]")
plt.xlabel("Time [ps]")
if save:
    plt.savefig("mss_series_PBC_dt%0.3f_W10E%i.svg" %(dt,Wpower))
plt.show()



g = np.sum(g_series, axis=0)/steps

fig, ax = plt.subplots()
plt.plot(mean_rad_array,g/(4*np.pi*mean_rad_array**2))
plt.ylabel(r"g(r)")
plt.xlabel(r"r [$\AA$]")
if save:
    plt.savefig("pair_corr_PBC_dt%0.3f_W10E%i.svg" %(dt,Wpower))
plt.show()

if save:
    np.savetxt("lam_PBC_dt%0.3f_W10E%i.csv" %(dt,Wpower),lam_series, delimiter=',')
    np.savetxt("energy_PBC_dt%0.3f_W10E%i.csv" %(dt,Wpower),energies, delimiter=',')
    np.savetxt("pressure_PBC_dt%0.3f_W10E%i.csv" %(dt,Wpower),pressures, delimiter=',')
    np.savetxt("mss_dt%0.3f_W10E%i.csv" %(dt,Wpower),mss_series, delimiter=',')
    np.savetxt("s_dt%0.3f_W10E%i.csv" %(dt,Wpower),s, delimiter=',')
    np.savetxt("vs_dt%0.3f_W10E%i.csv" %(dt,Wpower),vs, delimiter=',')
    

# %% animate the solution in 2D
skip = 10

s_series1 = s_series[::skip]

fig, ax = plt.subplots()
ax.set(xlim=(0, cell_l), ylim=((0, cell_l)))


dots = ax.scatter(s_series1[0][:,0], s_series1[0][:,1], c='pink')


def animate(i):
    dots.set_offsets(np.c_[s_series1[i][:, 0], s_series1[i][:, 1]])

anim = FuncAnimation(
    fig, animate, interval=1, frames=len(s_series1))

plt.draw()

plt.show()

